# pitchpipe

To install, grab the portaudio library first:

```
$ brew install portaudio
```

If needed, add some environment variables so that `pip` can find portaudio:

```
$ export CPATH=/usr/local/include
$ export LIBRARY_PATH=/usr/local/lib
```

Then let pip do the rest:

```
$ pip install -r requirements.txt
```
