import abc
import argparse
import asyncio
from asyncio.subprocess import DEVNULL, PIPE
from collections import defaultdict
import contextlib
import enum
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, \
                   Optional, Tuple

from gi.repository import GLib  # type: ignore
import gbulb  # type: ignore
import jinja2
import pydbus

import logutil
from logutil import APP_NAME, DEFAULT_LOG_LEVEL, LogMixin, init_log_levels_from_env

try:
    import aiobspwm
except ImportError:
    # just raise if bspwm features are used
    pass

try:
    import pulsectl
except ImportError:
    # required for optional pulseaudio widget
    pass

CACHE_DIR = Path(GLib.get_user_cache_dir()) / APP_NAME
CONFIG_DIR = Path(GLib.get_user_config_dir()) / APP_NAME

if not CACHE_DIR.exists():
    CACHE_DIR.mkdir()

if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir()

LOG_PATH = str(CACHE_DIR / f'{APP_NAME}.log')
logutil.init_logger(LOG_PATH, logging.getLogger())
log = logging.getLogger(APP_NAME)

sys_bus = pydbus.SystemBus()

RequestUpdate = Callable[[], Awaitable[None]]


class Widget(LogMixin, metaclass=abc.ABCMeta):
    """
    A text-based widget that can be placed on the panel
    """
    log_level = 'INFO'

    @abc.abstractmethod
    async def update(self) -> str:
        """
        Return a new value for what should be displayed
        """

    @abc.abstractmethod
    async def watch(self, request_update: RequestUpdate) -> None:
        """
        Run in a loop waiting on whatever condition, then call request_update
        when an update is desired.

        This is guaranteed to be called before update() is first called.
        This guarantee means that initialization that cannot be done in
        __init__ because there is no event loop can be done
        before the while loop here.

        Parameters:
        request_update -- *async* callable that requests up to the panel to
                          update this widget
        """
        while True:
            await request_update()
            await asyncio.sleep(1)


class StaticWidget(Widget):
    """
    A widget that displays a static value
    """
    def __init__(self, value: str) -> None:
        """
        Parameters:
        value -- value to display
        """
        super().__init__()
        self._value = value

    async def update(self) -> str:
        return self._value

    async def watch(self, request_update: RequestUpdate) -> None:
        # do nothing because this widget never changes
        await request_update()


class DateTimeWidget(Widget):
    """
    A widget that displays the date
    """
    def __init__(self, fmt: str, update: int = 1) -> None:
        """
        Parameters:
        fmt -- strftime() compatible format string. Example: '%b %-d %H:%M'
               Output: 'Jun 23 21:12'
        update -- update interval in seconds
        """
        super().__init__()
        self._format = fmt
        self._update = update

    async def update(self) -> str:
        time.tzset()
        return time.strftime(self._format)

    async def watch(self, request_update: RequestUpdate) -> None:
        while True:
            await request_update()
            await asyncio.sleep(self._update)


class BspwmWidget(Widget):
    """
    A widget for displaying bspwm status based on aiobspwm.
    """
    log_level = 'DEBUG'

    def __init__(self,
                 template: str,
                 sock: Optional[str] = None,
                 ctx: Dict[str, Any] = {}) -> None:
        """
        Parameters:
        template -- jinja2 template string for the widget's display
                    Context:
                    - wm (aiobspwm.WM object)
                    - ctx
        sock -- alternate bspwm socket path to use. Default is to find it
                automagically.
        ctx -- extra context to pass to templating stage (allows for
               parameterisation of the template)
        """
        super().__init__()
        if 'aiobspwm' not in globals():
            raise ImportError('aiobspwm is required for BspwmWidget, but '
                              'was not found')
        self._template = jinja2.Template(template, autoescape=False)
        self._sock = sock or aiobspwm.find_socket()
        self._ctx = ctx

    def format(self, wm: 'aiobspwm.WM') -> str:
        """
        Take a WM object and return a formatted-up version
        """
        if wm.focused_monitor == None:
            self.log.error('bspwm context: {}', wm)
            raise ValueError(
                'focused_monitor is None! This should never happen')
        return self._template.render(wm=wm, ctx=self._ctx)

    async def update(self) -> str:
        if not hasattr(self, '_wm'):
            # this is here to mitigate issues with the event loop not
            # being initialised on config load (and thus __init__)
            self._updated.set()
            self._wm = aiobspwm.WM(self._sock,
                                   evt_hook=lambda ln: self._updated.set())
            await self._wm.start()

            # store this in the object to avoid it getting GC'd
            self._wm_task = asyncio.create_task(self._wm.run())

            self._wm_initialized.set()
        await self._wm_initialized.wait()
        return self.format(self._wm)

    async def watch(self, request_update: RequestUpdate) -> None:
        self._updated = asyncio.Event()
        # this ensures that we don't try to format output when we don't have
        # the wm object initialized
        self._wm_initialized = asyncio.Event()
        while True:
            await request_update()
            await self._updated.wait()
            self._updated.clear()


class DBusPropertyChangeWatcher(LogMixin):
    """
    Object that keeps its properties up to date using PropertyChanged signals
    """
    def __init__(self,
                 bus_name: str,
                 object_path: str,
                 iface: str,
                 hook: Callable[[], None] = lambda: None) -> None:
        """
        Parameters:
        bus_name -- name of the service to use on the bus
        object_path -- path to the object which is being watched
        iface -- interface on that object to watch the properties of

        Optional Parameters:
        hook -- hook function to call on all changes
        """
        super().__init__()
        bus_obj = sys_bus.get(bus_name, object_path)
        self._state = bus_obj.GetAll(iface)
        sys_bus.subscribe(sender=bus_name,
                          iface='org.freedesktop.DBus.Properties',
                          object=object_path,
                          signal_fired=self.on_change)
        self._hook = hook

    def on_change(self, sender: str, obj: Any, iface: str, signal: str,
                  params: Tuple) -> None:
        prop_change_iface, prop_change_args, *rest = params
        self.log.debug('Prop change %s', prop_change_args)
        self._state.update(prop_change_args)
        self._hook()

    def __dir__(self) -> List[str]:
        return list(set(self._state.keys()) | set(super().__dir__()))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._state[name]
        except KeyError:
            raise AttributeError(f'Could not find the property {name!r}')


class UPowerState(enum.IntEnum):
    UNKNOWN = 0
    CHARGING = 1
    DISCHARGING = 2
    EMPTY = 3
    FULLY_CHARGED = 4
    PENDING_CHARGE = 5
    PENDING_DISCHARGE = 6


class UPowerDeviceType(enum.IntEnum):
    UNKNOWN = 0
    LINE_POWER = 1
    BATTERY = 2
    UPS = 3
    MONITOR = 4
    MOUSE = 5
    KEYBOARD = 6
    PDA = 7
    PHONE = 8


class UPowerWidget(Widget):
    """
    A widget for battery monitoring.

    Exposes the entire set of UPower information for a given device.
    """
    log_level = 'INFO'
    _updated: Optional[asyncio.Event]

    def __init__(self,
                 fmt: str,
                 device: str = 'DisplayDevice',
                 ctx: Any = None) -> None:
        """
        Parameters:
        fmt -- template string. Context:
               - device -- autoupdated object with the properties of the
                           DBus interface org.freedesktop.UPower.Device
               - State -- enum containing the meanings of the State
                          property
               - DeviceType -- enum containing the meanings of the Type
                               property
               - ctx -- custom context from the ctx argument of
                        this constructor
        device -- device name

        Optional Parameters:
        ctx -- context object passed verbatim to the template
        """
        super().__init__()
        self.up_dev = DBusPropertyChangeWatcher(
            bus_name='org.freedesktop.UPower',
            object_path=f'/org/freedesktop/UPower/devices/{device}',
            iface='org.freedesktop.UPower.Device',
            hook=self.on_change)
        self._template = jinja2.Template(fmt, autoescape=False)
        self._ctx = ctx
        self._updated = None

    def on_change(self):
        if self._updated:
            self._updated.set()

    async def update(self) -> str:
        return self._template.render(device=self.up_dev,
                                     State=UPowerState,
                                     DeviceType=UPowerDeviceType,
                                     ctx=self._ctx)

    async def watch(self, request_update: RequestUpdate) -> None:
        self._updated = asyncio.Event()
        while True:
            await request_update()
            await self._updated.wait()
            self.log.debug('UPower update')
            self._updated.clear()


class ServiceContainer(defaultdict):
    def __init__(self, from_dict: Dict = {}) -> None:
        super().__init__(dict, from_dict)

    def filter(self, prop: str, value: Any) -> Dict[str, Any]:
        return ServiceContainer({k: v for (k, v) in self.items() \
                                 if v.get(prop) == value})


class ConnmanServiceWatcher(ServiceContainer, LogMixin):
    """
    Object that keeps a list of connman services up to date using signals
    """
    log_level = 'INFO'

    def __init__(self,
                 hook: Callable[[], None] = lambda: None,
                 **kwargs) -> None:
        """
        Optional Parameters:
        hook -- hook function to call on all changes
        """
        super().__init__()
        self.reload()
        sys_bus.subscribe(sender='net.connman',
                          iface='net.connman.Manager',
                          object='/',
                          signal='ServicesChanged',
                          signal_fired=self._on_change)
        sys_bus.subscribe(sender='net.connman',
                          iface='net.connman.Service',
                          signal='PropertyChanged',
                          signal_fired=self._on_change)
        self._hook = hook

    def reload(self) -> None:
        super().clear()
        bus_obj = sys_bus.get('net.connman', '/')
        services = bus_obj.GetServices()
        for svc in services:
            super().__setitem__(svc[0], svc[1])

    def _on_change(self, sender: str, obj: str, iface: str, signal: str,
                   params: Tuple[str, Dict]) -> None:
        if signal == 'ServicesChanged':
            updates, deletes = params
            for update in updates:
                self.log.debug('Service update %s %s', update[0], update[1])
                super().__getitem__(update[0]).update(update[1])

            for delete in deletes:
                self.log.debug('Service delete %s', delete)
                with contextlib.suppress(KeyError):
                    super().__delitem__(delete)
        elif signal == 'PropertyChanged':
            name, val = params
            self.log.debug('Service %s %r = %r', obj, name, val)
            self.__getitem__(obj)[name] = val
        self._hook()


class ConnmanWidget(Widget):
    """
    Widget to show state of Connman connections
    """
    _updated: Optional[asyncio.Event]

    def __init__(self, fmt: str, ctx: Any = None) -> None:
        """
        Parameters:
        fmt -- jinja2 template string to render
               Context:
               - services: ConnmanServiceWatcher instance containing Connman
                           services on the system in a ServiceContainer
                           (extended dict with easy filtering) keyed by
                           service name
               - ctx: context object from ctx parameter

        Optional parameters:
        ctx -- context object, passed directly to template
        """
        super().__init__()
        self._services = ConnmanServiceWatcher(hook=self._on_change)
        self._template = jinja2.Template(fmt, autoescape=False)
        self._ctx = ctx
        self._updated = None

    def _on_change(self) -> None:
        if self._updated:
            self._updated.set()

    async def update(self) -> str:
        return self._template.render(services=self._services, ctx=self._ctx)

    async def watch(self, request_update: RequestUpdate) -> None:
        self._updated = asyncio.Event()
        while True:
            await request_update()
            await self._updated.wait()
            self._updated.clear()


class Event_ts(asyncio.Event):
    """
    A thread safe version of the asyncio Event

    NOTE: clear() is not thread safe

    Taken from https://stackoverflow.com/a/33006667
    """
    def set(self):
        # XXX: uses undocumented internal attribute, _loop, of Event
        self._loop.call_soon_threadsafe(super().set)  # type: ignore


class PulseStateWatcher(LogMixin):
    """
    A class to keep track of pulseaudio state.

    Usage:
    >>> psw = PulseStateWatcher()
    >>> psw.run()
    """
    default_sink: Optional[int]
    volume: Optional[float]
    mute: Optional[bool]
    done_init: Optional[Event_ts]

    def __init__(self,
                 done_init: Optional[Event_ts] = None,
                 update_hook: Callable[[], None] = lambda: None) -> None:
        """
        Keyword parameters:
        done_init -- event set when initialization is done
        update_hook -- called whenever there is a (potential) change in
                       Pulse state
        """
        self.pulse = pulsectl.Pulse(__name__)
        self._update_hook = update_hook
        self.done_init = done_init

    def _init(self):
        self._reload_default_sink()
        self._reload_volume()

    def __enter__(self):
        self.pulse.__enter__()
        self._init()
        return self

    def __exit__(self, *args):
        self.pulse.__exit__(None, None, None)

    def _event_cb(self, evt):
        # we can't actually do anything with events here since the
        # pulsectl loop is still running
        self.log.debug('event: type %r idx %r facility %r', evt.t, evt.index,
                       evt.facility)
        self.prev_evt = evt
        raise pulsectl.PulseLoopStop()

    def _find_default_sink(self):
        for _ in range(3):
            server_info = self.pulse.server_info()
            default_name = server_info.default_sink_name  # type: ignore
            sinks = self.pulse.sink_list()
            self.log.debug('Finding default sink, server info %r; sinks %r',
                           server_info, sinks)
            sink = next((x for x in sinks if x.name == default_name), None)
            if not sink:
                self.log.error(
                    'Failed to find default sink: server info %r; sinks: %r',
                    server_info, sinks)
                time.sleep(1)
                continue
            return sink.index
        self.default_sink = None

    def _reload_default_sink(self):
        self.log.debug('Reloading sinks')
        self.default_sink = self._find_default_sink()

    def _reload_volume(self):
        if not self.default_sink:
            self._reload_default_sink()
            if not self.default_sink:
                return

        try:
            sink = self.pulse.sink_info(self.default_sink)
        except pulsectl.PulseIndexError:
            self.log.exception(
                'Index error while fetching volume, wrong default sink?')
            self._reload_default_sink()
            return

        self.volume = sink.volume.value_flat  # type: ignore
        self.mute = sink.mute  # type: ignore

    def _handle_event(self, evt):
        if evt.facility == 'server':
            # might be a default sink change, reload that
            prev_default = self.default_sink
            self._reload_default_sink()
            if self.default_sink != prev_default:
                self._reload_volume()
        elif evt.facility == 'sink' and evt.t in [
                pulsectl.PulseEventTypeEnum.new,  # type: ignore
                pulsectl.PulseEventTypeEnum.remove  # type: ignore
        ]:
            # changed sink list; default sink may have changed
            self._reload_default_sink()
        elif evt.facility == 'sink':
            # might be a volume change, reload volume
            self._reload_volume()
        if self.volume:
            self.log.debug('Volume of sink %s: %s', self.default_sink,
                           round(self.volume * 100))

    def run(self):
        """
        Run an event loop and keep the state on this class updated
        """
        with self:
            if self.done_init:
                self.done_init.set()
            self.log.debug('Starting pulse state watcher')
            while True:
                self.pulse.event_mask_set('sink', 'server')
                self.pulse.event_callback_set(self._event_cb)

                self.pulse.event_listen()  # blocks until event received
                self._handle_event(self.prev_evt)
                self._update_hook()


class PulseAudioWidget(Widget):
    """
    Widget to display the PulseAudio volume
    """
    def __init__(self, fmt: str, ctx: Any = None) -> None:
        """
        Parameters:
        fmt -- jinja2 template to render as output
               Context:
               - state: PulseStateWatcher of the current state
               - ctx: context parameter as passed to constructor
        ctx -- passed directly to template rendering
        """
        super().__init__()
        self._template = jinja2.Template(fmt, autoescape=False)
        self._ctx = ctx

    def _on_change(self):
        self._updated.set()

    async def update(self):
        return self._template.render(state=self._state, ctx=self._ctx)

    async def watch(self, request_update: RequestUpdate):
        self._updated = Event_ts()
        watcher_done_init = Event_ts()
        self._state = PulseStateWatcher(done_init=watcher_done_init,
                                        update_hook=self._on_change)
        threading.Thread(target=self._state.run, daemon=True).start()
        await watcher_done_init.wait()
        while True:
            await request_update()
            await self._updated.wait()
            self._updated.clear()


class SubprocessWidget(Widget):
    """
    A widget that takes lines on stdin and displays them on the panel.
    This is ideal for xtitle.
    """
    def __init__(self,
                 cmd: List[str],
                 fmt: str = "{{ line|trim }}",
                 ctx: Any = None) -> None:
        """
        Parameters:
        cmd -- [exe_location, arguments...] of process to execute

        Optional parameters:
        fmt -- jinja2 template to render as output
               Context:
               - line: latest line from subprocess stdout
               - ctx: provided context parameter
        ctx -- passed directly to template rendering
        """
        super().__init__()
        self._cmd = cmd
        self._template = jinja2.Template(fmt, autoescape=False)
        self._ctx = ctx
        self._last_line = ''

    async def update(self):
        return self._template.render(line=self._last_line, ctx=self._ctx)

    async def watch(self, request_update: RequestUpdate):
        proc = await asyncio.create_subprocess_exec(*self._cmd,
                                                    stdout=PIPE,
                                                    stderr=DEVNULL)
        assert proc.stdout
        while True:
            await request_update()
            self._last_line = (await proc.stdout.readline()).decode()


class PanelAdapter(metaclass=abc.ABCMeta):
    """
    Make the panel actually display somewhere
    """
    @abc.abstractmethod
    async def write(self, panel_value: str) -> None:
        """
        Write the specified value to the panel

        Parameters:
        panel_value -- value to write
        """


class SubprocessAdapter(PanelAdapter):
    """
    A PanelAdapter that sends newline terminated lines to the stdin
    of a process
    """
    def __init__(self, proc: List[str]) -> None:
        """
        Parameters:
        proc -- [exe_location, arguments...] of process to execute
        """
        self._proc_args = proc

    def _create_proc(self) -> Coroutine[Any, Any, asyncio.subprocess.Process]:
        return asyncio.create_subprocess_exec(*self._proc_args,
                                              stdin=PIPE,
                                              stdout=DEVNULL,
                                              stderr=DEVNULL)

    async def write(self, panel_value: str) -> None:
        if not hasattr(self,
                       '_process'):  # or self._process.returncode is not None:
            log.info('Creating panel subprocess')
            self._process = await self._create_proc()

        assert self._process.stdin
        try:
            self._process.stdin.write((panel_value + '\n').encode('utf-8'))
            await self._process.stdin.drain()
        except (GLib.GError, ConnectionResetError):
            log.exception('Failed to write to subprocess adapter')
            self._process.stdin.close()
            await self._process.wait()
            # this stops a GLib.GError: g-io-channel-error-quark: Bad file descriptor (8)
            # replacing the variable should be exactly the same thing but it isn't
            del self._process


class StdoutAdapter(PanelAdapter):
    """
    A PanelAdapter that spits lines out on stdout
    """
    async def write(self, panel_value):
        print(panel_value)


FormattingPosition = str
WidgetDict = Dict[FormattingPosition, List[Widget]]


class Panel:
    def __init__(self, widgets: WidgetDict, out_fmt: str,
                 out_adapter: PanelAdapter) -> None:
        """
        Parameters:
        widgets -- position: list of widgets structure
        out_fmt -- jinja2 template for output. Widget values are passed in as
                   {widgets_key: joined_widgets}
        out_adapter -- PanelAdapter to output to
        """
        self._widgets = widgets
        self._widget_state: Dict[Widget, str] = {}
        self._out_fmt = jinja2.Template(out_fmt, autoescape=False)
        self._adapter = out_adapter
        # NOTE: please don't run widgets on separate threads; use IPC
        #       to run the widget on the main thread and exchange data
        self._need_redraw = asyncio.Event()
        self._widget_tasks = []

    async def redraw(self) -> None:
        await self._adapter.write(self._draw())

    def _draw(self) -> str:
        """
        Format the output for the panel.

        Returns:
        String representing the new value
        """
        out = {}
        for (pos, widgets) in self._widgets.items():
            out[pos] = ''.join(self._widget_state.get(w, '') for w in widgets)
        return self._out_fmt.render(out)

    def _start_widget(self, widget: Widget) -> asyncio.Task:
        """
        Starts a widget's watch routine

        Parameters:
        widget -- widget to start up
        """
        log.debug('Starting %r', widget)

        async def request_update() -> None:
            # log.debug('widget %s requested update', widget)
            self._widget_state[widget] = await widget.update()
            self._need_redraw.set()

        return asyncio.create_task(widget.watch(request_update))

    async def _init_widgets(self) -> None:
        for widget_list in self._widgets.values():
            log.debug('initializing %r', widget_list)
            self._widget_tasks.extend(
                self._start_widget(w) for w in widget_list)
        log.info('Widgets initialized')

    async def run(self) -> None:
        await self._init_widgets()

        while True:
            # Widgets will request updates, the updates will be executed in
            # parallel, then the bar will be flagged for redraw at some point
            # in the future. The purpose of this design is to allow multiple
            # updates to occur per redraw cycle since updates are cheaper than
            # redraws (lemonbar is slow!)
            await self._need_redraw.wait()
            self._need_redraw.clear()
            await self.redraw()


def start(cfg: dict) -> None:
    """
    Start a panel given a config dict

    Parameters:
    cfg -- config dict
    """
    gbulb.install(gtk=False)
    loop = asyncio.get_event_loop()
    panel = Panel(cfg['widgets'], cfg['out_fmt'], cfg['out_adapter'])
    loop.create_task(panel.run())
    loop.run_forever()


CONFIG_REQUIRED = ['widgets', 'out_fmt', 'out_adapter']


def main(args: argparse.Namespace) -> None:
    import runpy
    cfg = runpy.run_path(str(args.config))
    for req in CONFIG_REQUIRED:
        if req not in cfg:
            raise ValueError(f'Config missing value {req!r}')
    log.setLevel(cfg.get('log_level', DEFAULT_LOG_LEVEL))
    init_log_levels_from_env()
    start(cfg)


def _parse_args(argv: List[str] = sys.argv[1:]) -> argparse.Namespace:
    default_config = CONFIG_DIR / f'config.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        '-c',
                        help='Config file to use',
                        type=Path,
                        default=default_config)
    return parser.parse_args(argv)


def entry_point() -> None:
    try:
        main(_parse_args())
    except Exception as e:
        log.exception('Got exception while running panel')
        raise


if __name__ == '__main__':
    entry_point()
