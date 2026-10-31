# Background change

## Install and run

```bash
pip install -r requirements.txt
python change_bg.py input.jpg --color "#1E90FF" -o out.png --save-mask
```

- `--color` takes 3- or 6-digit hex, with or without `#`.
- `-o` defaults to `<input name>_bg.png` next to the input; `--save-mask` also writes the alpha as `<output name>_mask.png`.
- `--device cpu` or `--device cuda` (default: cuda if available).
- The first run downloads BiRefNet (444 MB, cached by Hugging Face). On my laptop CPU a ~3 MP photo takes about 40 s.

## Results

## Decisions and why

## What still fails

## Related work

## How I used AI tools

## What I would do next
