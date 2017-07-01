import abc
import argparse
import asyncio
from asyncio.subprocess import DEVNULL, PIPE
import datetime
import enum
import logging
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable, Container, Dict, List, \
                   NamedTuple, Tuple

from gi.repository import GLib
import gbulb
import jinja2
import pydbus

try:
    import aiobspwm
except ImportError:
    # just raise if bspwm features are used
    pass


APP_NAME = 'aiopanel'
CACHE_DIR = Path(GLib.get_user_cache_dir()) / APP_NAME
CONFIG_DIR = Path(GLib.get_user_config_dir()) / APP_NAME

if not CACHE_DIR.exists():
    CACHE_DIR.mkdir()

if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir()

LOG_PATH = str(CACHE_DIR / f'{APP_NAME}.log')
DEFAULT_LOG_LEVEL = logging.INFO

log = logging.getLogger(APP_NAME)

# don't override settings in our logger put there before we loaded
if not log.level:
    log.setLevel(DEFAULT_LOG_LEVEL)

fmt = logging.Formatter(
    '{asctime} {levelname} {filename}:{lineno}: {message}',
    datefmt='%b %d %H:%M:%S',
    style='{'
)
# don't add handlers repeatedly when I use autoreload
for handler in log.handlers:
    if isinstance(handler, logging.FileHandler):
        break
else:
    hnd = logging.FileHandler(LOG_PATH)
    hnd.setFormatter(fmt)
    log.addHandler(hnd)

sys_bus = pydbus.SystemBus()


class UniqueQueue(asyncio.Queue):
    _queue: Container[Any]

    async def put_unique(self, item: Any) -> None:
        """
        Put an item on the queue, checking if it is already there

        Parameters:
        item -- item to put on queue
        """
        if item not in self._queue:
            await self.put(item)


RequestUpdate = Callable[[], Awaitable[None]]


class Widget(metaclass=abc.ABCMeta):
    """
    A text-based widget that can be placed on the panel
    """
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
        self._format = fmt
        self._update = update

    async def update(self) -> str:
        return datetime.datetime.strftime(datetime.datetime.now(),
                                          self._format)

    async def watch(self, request_update: RequestUpdate) -> None:
        while True:
            await request_update()
            await asyncio.sleep(self._update)


class BspwmWidget(Widget):
    """
    A widget for displaying bspwm status based on aiobspwm.
    """
    def __init__(self, template: str, sock: str = None,
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
        return self._template.render(wm=wm, ctx=self._ctx)

    async def update(self) -> str:
        if not hasattr(self, '_wm'):
            # this is here to mitigate issues with the event loop not
            # being initialised on config load (and thus __init__)
            self._updated.set()
            self._wm = aiobspwm.WM(self._sock,
                    evt_hook=lambda ln: self._updated.set())
            await self._wm.start()
            asyncio.ensure_future(self._wm.run())
        return self.format(self._wm)

    async def watch(self, request_update: RequestUpdate) -> None:
        self._updated = asyncio.Event()
        while True:
            await request_update()
            await self._updated.wait()
            self._updated.clear()


class DBusPropertyChangeWatcher:
    """
    Object that keeps its properties up to date using PropertyChanged signals
    """
    def __init__(self, bus_name: str, object_path: str, iface: str,
                 hook: Callable[[], None] = lambda: None) -> None:
        """
        Parameters:
        bus_name -- name of the service to use on the bus
        object_path -- path to the object which is being watched
        iface -- interface on that object to watch the properties of

        Optional Parameters:
        hook -- hook function to call on all changes
        """
        bus_obj = sys_bus.get(bus_name, object_path)
        self._state = bus_obj.GetAll(iface)
        sys_bus.subscribe(sender=bus_name,
                          iface='org.freedesktop.DBus.Properties',
                          object=object_path,
                          signal_fired=self.on_change)
        self._hook = hook

    def on_change(self, sender: str, obj: Any, iface: str, signal: str,
                  params: Tuple[str, Dict]) -> None:
        prop_change_iface, prop_change_args, *rest = params
        log.debug('Prop change %s', prop_change_args)
        self._state.update(prop_change_args)
        self._hook()

    def __dir__(self) -> List[str]:
        return set(self._state.keys()) | set(super().__dir__())

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
    def __init__(self, fmt: str, device: str = 'DisplayDevice',
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
        self.up_dev = DBusPropertyChangeWatcher(
            bus_name='org.freedesktop.UPower',
            object_path=f'/org/freedesktop/UPower/devices/{device}',
            iface='org.freedesktop.UPower.Device',
            hook=self.on_change)
        self._template = jinja2.Template(fmt)
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
            log.debug('UPower update')
            self._updated.clear()


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
        self._process_coro = asyncio.create_subprocess_exec(
                *proc,
                stdin=PIPE,
                stdout=DEVNULL,
                stderr=DEVNULL
        )

    async def write(self, panel_value):
        if not hasattr(self, '_process'):
            self._process = await self._process_coro
        self._process.stdin.write((panel_value + '\n').encode('utf-8'))
        await self._process.stdin.drain()


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
        self._update_queue = UniqueQueue()

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

    def _start_widget(self, widget: Widget):
        """
        Starts a widget's watch routine

        Parameters:
        widget -- widget to start up
        """
        async def request_update() -> None:
            # log.debug('widget %s requested update', widget)
            await self._update_queue.put_unique(widget)
        asyncio.ensure_future(widget.watch(request_update))

    async def _init_widgets(self) -> None:
        for (pos, widgets) in self._widgets.items():
            for widget in widgets:
                self._start_widget(widget)
        log.info('Widgets initialized')

    async def run(self) -> None:
        await self._init_widgets()
        while True:
            widget = await self._update_queue.get()
            # XXX: this is effectively a blocking operation and should
            #      somehow be concurrent-ised
            self._widget_state[widget] = await widget.update()
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
    asyncio.ensure_future(panel.run())
    loop.run_forever()


CONFIG_REQUIRED = [
    'widgets',
    'out_fmt',
    'out_adapter'
]


def main(args: argparse.Namespace) -> None:
    import runpy
    cfg = runpy.run_path(str(args.config))
    for req in CONFIG_REQUIRED:
        if req not in cfg:
            raise ValueError(f'Config missing value {req!r}')
    log.setLevel(cfg.get('log_level', DEFAULT_LOG_LEVEL))
    start(cfg)


def _parse_args(argv: List[str] = sys.argv[1:]) -> argparse.Namespace:
    default_config = CONFIG_DIR / f'config.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
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
