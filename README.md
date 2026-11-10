# Zumzum

Zumzum is a small language for audio repair: each fix is a short program that is checked, run and then measured.

![A Zumzum program, with an 1888 wax cylinder recording before and after it](docs/img/readme_program.png)

## Install

You need Python 3.12 or later. From the repository folder (on Windows, activate with `venv\Scripts\activate`):
```
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

To work on your own files you also need [ffmpeg](https://ffmpeg.org/) on your PATH. It is not included.

## Example

Bring a recording to a delivery target, then check the result:
```
python tools/fix.py --list                                  # the delivery targets and their numbers
python tools/fix.py --spec podcast-apple talk.wav out/      # or audiobook-acx, ebu-r128, atsc-a85
python tools/verify.py out/talk.mp3 --source talk.wav       # check the repair record
```

This writes `out/talk.mp3` (the delivered file), `talk.zz` (the program that ran), `talk_sheet.png` (a sheet with
spectrograms before and after), `talk.json` (the numbers before and after) and a repair record that
`tools/verify.py` can check. The exit code is 0 only when the delivered file meets the targets.

## What it does

- Removes or reduces hum, hiss, clicks and short tones such as phone dialling.
- Turns drums down under other music.
- Brings a file to a delivery loudness: Apple Podcasts, ACX, EBU R 128 or ATSC A/85. Results are measured against
  each specification's published numbers. This is not a certification. Not affiliated with or endorsed by Apple,
  Audible (ACX), the EBU or ATSC; targets are taken from their published specifications.
- Can turn sound down only (`attenuate`): everything you hear comes from your recording; it can only lower parts
  of it, never add sound.
- Puts a repair record in every output: the program, each check's result, and hashes of the source and the result.

## What it doesn't do

- It does not limit or compress. A file whose peaks leave too little room is reported as "needs limiting", not
  fixed. On our 40 public test recordings, 26 met their targets: all 20 broadcast files, 5 of 10 podcast files
  and 1 of 10 audiobook files.
- It does not remove noise when fixing loudness. A noisy recording keeps its noise.
- Its checks are measurements, not a judgement of how the result sounds. Listen to the result.
- It is a prototype: the language may change, and there is one engine, which runs on the CPU.

## Read more

- [docs/guide.md](docs/guide.md): the full walkthrough (the language, the examples, asking in words).
- [docs/spec.md](docs/spec.md): the language definition.
- [CHANGELOG.md](CHANGELOG.md): what changed, and when.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The Apache licence does not license the name "Zumzum";
if you publish a changed version, please give it another name.
