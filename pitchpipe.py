import six
from six.moves import input

import math
from six.moves import queue
import threading
import time
import wave

import numpy
import pyaudio


class AudioPlayer(object):
    def __init__(self, sample_size, channels, rate):
        self.sample_size = sample_size
        self.channels = channels
        self.rate = rate
        self.p = pyaudio.PyAudio()
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def skip(self, frames):
        self.queue = queue.Queue()

    def write(self, data):
        self.queue.put(data)

    def close(self):
        self.queue = None
        self.thread.join(timeout=1.0)
        try:
            self.p.terminate()
        except Exception:
            pass

    def run(self):
        stream = self.p.open(format=self.p.get_format_from_width(self.sample_size),
                             channels=self.channels, rate=self.rate, output=True)
        while True:
            try:
                stream.write(self.queue.get())
            except AttributeError:
                break
        stream.stop_stream()
        stream.close()


class PitchAnalyzer(object):
    def __init__(self, sample_size, channels, rate):
        self.dtype = {1: numpy.int8, 2: numpy.int16}[sample_size]
        self.channels = channels
        self.rate = float(rate)
        self.frame_size = sample_size * channels
        self.frames_played = 0

        self.threshold = 0.6 * 2**(sample_size * 8) * channels
        # sigma = 1.0, kernel size = 5
        self.gaussian_kernel = numpy.array([0.06136, 0.24477, 0.38774, 0.24477, 0.06136])

        CHUNK_LEN = 4096
        self.base_freq = rate / CHUNK_LEN  # 10.7 Hz
        self.rfft_len = CHUNK_LEN / 2 + 1
        self.min_idx = int(146.8 / self.base_freq)  # with scordatura, lowest note is D3
        self.max_idx = int(4700  / self.base_freq)  # with harmonics, highest note is approx D8

    def skip(self, frames):
        self.frames_played += frames

    def write(self, data):
        tstamp = '%06.2f' % (self.frames_played / self.rate)
        nframes = len(data) / self.frame_size
        self.frames_played += nframes

        mags = self.sum_rffts(data)
        mags[:self.min_idx] = 0
        mags[self.max_idx:] = 0
        max_idxs = numpy.argpartition(mags, -20)[-20:]
        scores = sorted((numpy.sum(mags[idx : self.rfft_len : idx]) * idx**-0.4, idx)
                        for idx in max_idxs)

        best = scores[-1]
        if best[0] > self.threshold:
            freq = best[1] * self.base_freq
            semitones = math.log(freq / 440.0) / (math.log(2) / 12)

            semitones = int(round(semitones))
            pc_name = ['a', 'bb', 'b', 'c', 'c#', 'd', 'eb', 'e', 'f', 'f#', 'g', 'g#'][semitones % 12]

            args = [' '*(semitones + 15), pc_name]
            #args.append([s for s in scores[-3:]])
            #args.append(round(best[0]))
            #args.append(freq)
            six.print_(tstamp, *args)
        else:
            six.print_(tstamp)
        #amax = numpy.amax(mags)
        #for (i, mag) in enumerate(mags):
        #    #if i % 2: continue
        #    print i, "#"*int(100 * mag / amax)
        #    if i > 1000: break

    def sum_rffts(self, data):
        """De-interleave channels from data, perform rfft on each channel,
           and return the sum vector of the magnitudes of each rfft
        """
        data = numpy.frombuffer(data, dtype=self.dtype)  # no copy, read-only
        sum_mags = None
        mags = None
        for i in range(self.channels):
            clip = data[i::self.channels]
            rfft = numpy.fft.rfft(clip)
            if mags is None:
                mags = numpy.absolute(rfft)
            else:
                # re-use mags as much as possible
                numpy.absolute(rfft, out=mags)
            mags = numpy.convolve(mags, self.gaussian_kernel, mode='same')
            if sum_mags is None:
                sum_mags = mags.copy()
            else:
                sum_mags += mags
        return sum_mags


class Runner(object):
    def __init__(self):
        self.commands = queue.Queue()

    def play(self, wav, sinks):
        frames_played = 0
        t0 = time.time()
        rate = wav.getframerate()
        sample_rate = float(rate)
        skip_frames = {'n': 5*rate, 'N': 15*rate}
        for sink in sinks:
            sink.worktime = 0

        CHUNK_LEN = 4096
        frame_size = wav.getnchannels() * CHUNK_LEN * wav.getsampwidth()
        while True:
            try:
                cmd = self.commands.get_nowait()
                if cmd in ('n', 'N'):
                    nframes = skip_frames[cmd]
                    wav.readframes(nframes)
                    for sink in sinks:
                        if getattr(sink, 'skip', None):
                            sink.skip(nframes)
            except queue.Empty:
                pass
            data = wav.readframes(CHUNK_LEN)
            if len(data) != frame_size:
                break

            for sink in sinks:
                start = time.time()
                sink.write(data)
                sink.worktime += (time.time() - start)

            frames_played += CHUNK_LEN
            next_wake = t0 + frames_played / sample_rate
            sleep_dur = next_wake - time.time()
            if sleep_dur > 0.001:
                time.sleep(sleep_dur)

    def start_input(self):
        thread = threading.Thread(target=self.read_input)
        thread.daemon = True
        thread.start()

    def read_input(self):
        while True:
            x = input().strip()
            if x:
                self.commands.put(x)


def main(wavpath):
    wav = wave.open(wavpath, 'rb')
    NCHANNELS = wav.getnchannels()
    SAMPLE_SIZE = wav.getsampwidth()
    SAMPLE_RATE = wav.getframerate()
    assert SAMPLE_RATE == 44100

    runner = Runner()
    sinks = [sinktype(SAMPLE_SIZE, NCHANNELS, SAMPLE_RATE)
             for sinktype in (AudioPlayer, PitchAnalyzer)]

    start = time.time()
    try:
        runner.play(wav, sinks)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start
        for sink in sinks:
            try:
                six.print_("Average load for %s: %.2f%%" % (sink.__class__.__name__, 100 * sink.worktime/elapsed))
                sink.close()
            except Exception:
                pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="path to a WAV file")
    args = parser.parse_args()
    main(args.file)
