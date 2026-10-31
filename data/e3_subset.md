# E3 shadow-test set

Built by `python data/make_e3_subset.py` (needs `data/e1/` and `data/e2/`, built by `make_e1_subset.py` and
`make_e2_subset.py`; about 1 min once DESOBA is downloaded).
Output: `data/e3/<group>/...` (not committed, 370 MB) and `data/e3_subset.csv` (the list the notebook reads).
The first run downloads DESOBA (one 1.86 GB zip) to `data/DESOBA/` and keeps it, so a rebuild does not
download it again; the zip is read in place, nothing else is extracted. Delete it once the set is built
if disk space matters.

```
data/e3/
  A_synthetic/im/       225 PNG inputs: <portrait>_<background>_<shadow>.png
  A_synthetic/fg/, gt/  the 15 portraits and their mattes, resized to 1024 px (the true F and alpha)
  A_synthetic/lit/      45 lit backgrounds L: <portrait>_<background>.png
  A_synthetic/shadow/   60 true shadow maps T (16-bit): <portrait>_<shadow>.png
  B_desoba/{im,gt,lit,shadow,other}/   34 photos, object mask, L, T (16-bit), other shadows
  D_real/im/            9 photos (6 copied from E1, 3 downloaded)
```

## What a row gives

To score "keep the shadow", each A and B row gives the input, the subject's alpha (`gt`), the lit
background `lit` (L: the same scene without the subject's shadow) and the true shadow factor `shadow`
(T, 1 = lit, read with `cv2.imread(path, cv2.IMREAD_UNCHANGED) / 65535`; empty = no shadow, T = 1).
The right output on any new colour N is then known:

    answer = alpha * F + (1 - alpha) * T * N

with F = `fg` (A) or the photo itself (B, binary masks). D rows have no ground truth (judged by eye).

| Group | Rows | From | Ground truth | What it measures |
|---|---:|---|---|---|
| A_synthetic | 225 | the 15 E2 portraits × 3 backgrounds × 5 shadows | exact T and L | shadow-map and colour error, factor by factor (background, shadow length, softness) |
| B_desoba | 34 | DESOBA test, images with one object and its shadow | T from a photo with the shadow removed by hand | real shadows: real soft edges, coloured light, textured ground |
| D_real | 9 | 6 E1 photos + 3 Wikimedia Commons photos | none | studio fashion photos, by eye |

No public set has a known answer on studio fashion photos (checked: ISTD and ISTD+ have no subject
mask, SOBA has no shadow-free photo, DESOBAv2's shadow-free photos come from an inpainting model).
So A gives the controlled numbers, B checks them on real shadows, and D shows the target use.

## A_synthetic: a test with a known answer

Each portrait is the foreground F with its matte α, on a lit background L, with a shadow T cast by its
own silhouette: `input = α·F + (1 − α)·T·L + noise`. The right answer on N is `α·F + (1 − α)·T·N`.
The answer drops L: the brief asks for the subject's shadow, not for the old lighting.

| Factor | Values | Why |
|---|---|---|
| L | `white` (240), `dark` (64), `gradient` (240 at the top to 150 at the bottom) | white studio; dark studio, where the shadow is weak against the noise; light falloff that is not a shadow (like D07, D10) |
| Shadow length | 2 % and 12 % of the image diagonal | hugging the subject; longer than the 10 % margin of the inpainting method |
| Shadow softness | blur σ = 0.3 % and 2 % of the diagonal | hard edge (tests computing the shadow at 512 px); soft studio shadow |
| No shadow | T = 1 | the control: the background must stay plain |

- **Shadow**: `T = 1 − 0.5 · blur(shift(α))`. The core gets half the light, and the blurred edge covers
  every T from 0.5 to 1. One direction per portrait (seed 0): left or right, 0 to 30° down, the same for
  its four shadows; written in the `note` column. The shift replicates the image border, because the
  portraits are cut by the frame and the body (and its shadow) go on past it.
- **Noise**: Gaussian, σ = 2 grey levels, on the whole input, like sensor noise. Without it, every
  method would see a perfectly flat background.
- **Size**: 1024 px on the long side (BiRefNet predicts at 1024 px anyway). PNG, not JPEG, so the answer
  stays exact.
- **How much shadow** (white background, mean over the 15 portraits): background pixels with T < 0.98
  are 5.3 % (short hard), 16.6 % (short soft), 18.8 % (long hard) and 25.1 % (long soft) of the
  background. The darkest T on the background is 0.50 for the hard shadows, 0.55 (short soft) and
  0.51 (long soft), on average.

Check after the build, on all 225 inputs: input minus the formula has mean 0.000, standard deviation
2.02 and mean absolute value 1.58 grey levels, which is the added noise (σ = 2) and rounding.

Caveats:
- As in E2, in the hair a PPM-100 photo already contains its original background, so F there is not the
  true colour of the hair (the dark halo around the hair on white). The test stays exact, because the
  answer uses the same F; and the shadow metrics are taken on background pixels (α < 0.02).
- The shadow is a moved silhouette on a wall behind the subject. The PPM-100 portraits are cut at the
  waist, so there is no floor shadow with a known answer here; B and D have floor shadows.
- The shadow is grey (the same T on R, G and B), which favours a luminance method; B has coloured shadows.

## B_desoba: real shadows with a known answer

[DESOBA](https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBA) (Hong, Niu, Zhang, Zhang,
*Shadow Generation for Composite Image in Real-world Scenes*, AAAI 2022, arXiv 2104.10338) is SOBA
(Wang et al., CVPR 2020) with every shadow removed by professional photo editors, plus masks that pair
each object with its shadow. For each photo: `ShadowImage` (the photo), `DeshadowedImage` (the edit),
`InstanceMask` and `ShadowMask` (a shadow has the grey value of its object; 255 = a shadow whose object is
not annotated).

- **Selection**: the test images (`Testing_labels.txt`) whose instance mask has one value, DESOBA's own
  "BOS-free" rule (one object and its shadow, no other object-shadow pair): 34 images, as in their README.
- **L**: the shadow-free edit within 20 px of the object's shadow mask, the photo everywhere else (DESOBA
  builds its composites the same way). Other shadows stay in L, so the answer drops them: the brief asks
  for the subject's shadow only. Why 20 px: the soft edge of a shadow goes past its mask, and on 32 of the
  34 images 98 to 100 % of the editors' change lies within 20 px of it (measured before the build). The two
  exceptions: web-shadow0078 (67 %: editors also changed areas outside every mask) and lssd1822 (97 %).
- **T**: `clip(Y(photo) / Y(L), 0, 1)` with Rec. 601 luminance, exactly 1 wherever L is the photo, and 1
  on the object.
- **Content**: outdoor scenes (people, cars, animals, a few objects), 359 to 1600 px wide. The subject's
  shadow (T < 0.98) covers 0.6 to 31.7 % of the image (median 4.4 %), with a mean T of 0.17 to 0.66 in it
  (median 0.42). 10 of 34 images also have other shadows (`other/`), up to 48 % of the image (web-shadow0078,
  tree shade); the notebook can score with or without those pixels.

Caveats:
- The shadow-free photo is a manual edit; the DESOBA authors write that it "may not be perfectly accurate".
  Inside large shadows on textured ground, T shows some of that texture.
- T is a luminance ratio, so this answer ignores the colour of the shadow (often bluish outdoors). The
  per-channel ratio can be computed from `image` and `lit` if needed.
- Masks are binary (SOBA annotations): no soft edges, so B says nothing about hair.
- Not fashion, and most backgrounds are textured: B is a hard case for the methods, not Modelia's typical input.

Licence: the DESOBA repository is MIT, but the images come from ADE20K, SBU, ISTD, COCO and web searches
(SOBA paper, section 3), so their licences are mixed and partly non-commercial (ISTD). Evaluation only:
not committed, and no figure with these images is committed.

## D_real: fashion photos, by eye

| Image | Why | Licence | Shown in figures? |
|---|---|---|---|
| D02 white sneaker | soft contact shadow on white | CC BY-SA 4.0 | yes |
| D06 logo t-shirt | flat garment, faint shadow | CC BY-SA 4.0 | yes |
| D07 hanger shirt | light falloff on a dark blue background, and a cast shadow | CC BY 2.0 | yes |
| D10 low light | dark vignette, not a shadow: false-shadow check | unknown | **no** |
| D05 jump, white studio | control: almost no shadow, the output should not change | CC BY 2.0 | yes |
| D04 lace dress | textured ground (leaves, soil): expected failure | CC BY-SA 2.0 | yes |
| D11 studio soft shadow | full-body model on a white cyclorama; soft shadow on the floor and the wall | CC BY 2.0 | yes |
| D12 full body, grey studio | full-body on-model photo; contact shadows at the feet; light falloff on the wall | Public domain | yes |
| D13 belt drop shadow | garment accessory on light grey; soft drop shadow | Public domain | yes |

D11 to D13 were found by searching Wikimedia Commons ("Photo shoots", "People with white background",
"Clothing on white background") for a visible cast shadow; authors and links are in `data/e3_sources.csv`,
the E1 ones in `data/e1_fashion_sources.csv`. D11 is small (712 × 750); D12 is the 1920 px wide copy of a
4128 × 6192 original.
