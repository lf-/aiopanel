# aiopanel

An `asyncio` text-based panel system.

## Features

- WM and bar agnostic. By default, lemonbar-like bars that take lines on stdin
  and display them are supported, though there is a generic API that can be
  used to hook the output to anything that takes a string.
- Uses very little CPU time when running
- Rich customization through a Python based configuration file
- Concurrent/asynchronous programming model ensures all widgets are as
  responsive as possible

## Requirements

- Python 3.6+ (`f`-strings and `asyncio` changes)
- jinja2 (all templating)
- gbulb + GLib (event loop, will be used for DBus widget subscriptions)
- aiobspwm (optional, for bspwm widget)

## How to use

Config file is at `~/.config/aiopanel/config.py`<br/>
Log file is at `~/.cache/aiopanel/aiopanel.log`

Example config for a `bspwm`/`lemonbar` setup:

```python
import aiopanel

# An instance of a PanelAdapter subclass that defines where the panel
# renders to
out_adapter = aiopanel.SubprocessAdapter([
    'lemonbar',
    '-fSource Sans Pro:size=11',
    '-fFontAwesome:size=11',
    '-gx20'
])

# A jinja2 format template string for the whole panel. Gets the rendered
# output of whatever is in `widgets` as input
out_fmt = '{{ bspwm }}%{r}{{ date }}'

# A string or integer log level for the standard library logging module
# Optional.
log_level = 'DEBUG'

# these are just put here for convenience; they aren't config settings
bspwm_ctx = {
    'active_colour': '#ff4c5399',
    'inactive_colour': '#ff303030'
}

# same here
bspwm_template = """
{%- for m in wm.monitors.values() -%}
    %{B{{ ctx.active_colour if wm.focused_monitor == m else ctx.inactive_colour }}}
    {{- ' ' }}{{ m.name }}{{ ' ' }}%{B-}
    {%- for d in m.desktops.values() -%}
        %{B{{ ctx.active_colour if m.focused_desktop == d else ctx.inactive_colour }}}
        {{- ' ' }}{{ d.name }}{{ ' ' }}%{B-}
    {%- endfor -%}
{%- endfor -%}
"""

# position: List[Widget] dictionary of widgets.
# Rendered, concatenated (within the lists) output is the context for out_fmt
widgets = {
    'bspwm': [aiopanel.BspwmWidget(bspwm_template, ctx=bspwm_ctx)],
    'date': [aiopanel.DateTimeWidget('%b %-d %H:%M', update=2)]
}

```

## Writing a widget

A widget is expected to have two async methods on it:

`update()`, returning a str representing the desired output of the widget. This
is called whenever `watch()` asks for an update.

`watch(request_update)`, returning nothing. This runs in an infinite loop
awaiting a given update condition, then calling `request_update()`.

#### Note:

It generally makes more sense to call `request_update()` before waiting,
so that the initial loading of the widget happens immediately.
