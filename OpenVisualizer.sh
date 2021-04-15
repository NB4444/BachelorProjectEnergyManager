#!/bin/bash

# Open the visualizer in the browser
url="localhost:8888"
if which xdg-open >/dev/null; then
  xdg-open $url
elif which gnome-open >/dev/null; then
  gnome-open $url
fi
