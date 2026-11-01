# Background change

## Install and run

```bash
pip install -r requirements.txt
python change_bg.py input.jpg --color "#1E90FF" -o out.png --save-mask
python change_bg.py input.jpg --color "#1E90FF" --keep-shadows --shadow-strength 0.8 -o out.png
```

- `--color` takes 3- or 6-digit hex, with or without `#`.
- `-o` defaults to `<input name>_bg.png` next to the input; `--save-mask` also writes the alpha as `<output name>_mask.png`.
- `--device cpu` or `--device cuda` (default: cuda if available).
- `--edges fg` (default): soft alpha + foreground colour estimation, so the old background does not leak into the hair (E2). `soft` and `hard` are kept for comparison.
- `--keep-shadows` keeps the subject's shadows on the new colour (E3): the new colour is darkened where the photo is darker than its estimated lit background. Off by default, because on a textured background (grass, cobblestones) the texture is kept too. `--shadow-strength` from 0 (no shadow) to 1 (as measured, the default). `--shadow-method inpaint` (default) ignores light falloff on the backdrop but cuts shadows that reach farther than 10 % of the image diagonal from the subject; `flat` keeps long shadows on a plain backdrop but takes light falloff for shadow.
- The first run downloads BiRefNet (444 MB, cached by Hugging Face) and compiles the foreground estimation (numba, cached on disk). On my laptop CPU a ~3 MP photo takes about 40 s.

## Results

## Decisions and why

## What still fails

## Related work

## How I used AI tools

## What I would do next
