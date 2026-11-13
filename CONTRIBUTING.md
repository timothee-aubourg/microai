# Working on microAI

This file is for anyone changing the course, people and coding agents alike.
Read it before you edit a chapter.

## What the course is

`docs/` is the website (GitHub Pages serves it from `main`, folder `/docs`).
Each chapter is one HTML page plus a Python script in `code/` that implements
the idea with numpy. The shared runtime is `docs/assets/micro.js` (page chrome,
widgets, and a GPT forward pass that runs the trained weights in the browser).
The chapter list and the top bar live in `M.chapters` in that file.

## How a chapter teaches

Chapters 1 and 2 are the reference for the house style. Each section follows
this order:

1. **Question.** The problem in plain words.
2. **Worked example by hand.** Something small enough to follow on paper.
3. **Diagram.** An inline SVG figure (`<figure class="fig">`) drawn with the
   page's own colours (`.bx`, `.bx-fwd`, `.ln`, `.t-muted` …), so it works in
   dark mode.
4. **Definition at first use.** Every new term gets a `<div class="def">` box
   with `<dfn>term</dfn>` where it first appears, and an entry in
   `docs/assets/glossary.js`. Later uses link to it:
   `<a class="term" href="glossary.html#key">term</a>`.
5. **Experiment.** A `<section class="bench">` with a "Try this" line, followed
   by a **"What you should have seen"** paragraph that explains the result.
6. **Formula,** introduced only after the idea, with each symbol explained.
7. **Code** from `code/`, with a sentence on what each function does.
8. **Recap.** A `<div class="recap">` with three to five bullets at the end.

Two more pieces are added to every chapter by the shared script, from the
table `M.lesson` in `docs/assets/micro.js` (one entry per chapter):

- **Guess first.** A ten-second question with three options, shown under the
  opening question, then the answer in one or two sentences. A wrong guess is
  the useful case. There is a plain score, kept in the reader's browser, and
  nothing else: no streaks, no badges.
- **One line to keep.** One sentence at the very end of the chapter.

A new chapter needs an entry in that table. The answer and the line to keep
must follow from the chapter's own text and numbers.

Chapters 3 to 19 do not yet have all of the reference style: the definition
boxes, glossary entries and recap of chapters 1 and 2 are still to be added
there. Chapters 20 to 23 have it.

Optional maths goes in a folded `<details class="refresher">`. Step-by-step
animations use `M.stepper` (see the BPE walkthrough in chapter 2).

Every number quoted in the text must come from running the code or the data,
not from memory. If a widget's behaviour is described, check it in the browser.

The course does not credit named individuals as inspiration.

## How a page looks

The look is paper and ink: ivory paper and brown-black ink by day, dark paper
and cream ink by night. It is defined once, as variables at the top of
`docs/assets/style.css`.

- **Three colours carry meaning** and mean the same on every page: ultramarine
  (`--fwd`) for values flowing forward and for the selected thing, raspberry
  (`--grad`) for gradients and errors, amber (`--prob`, `--prob-fill`) for
  probabilities. Never write a colour value in a page: use the variables, and
  `M.css('--fwd')` inside a canvas, so both themes follow.
- **Type.** Bodoni Moda (`--display`) for the chapter title, the opening
  question and the line to keep. Libre Franklin (`--sans`) for everything else.
  Both are in `docs/assets/fonts/` under the SIL Open Font License.
- **Section headings** get a short brush stroke from the stylesheet. Do not
  add rules or borders by hand.
- **Icons are a dab and a gesture.** Every icon is one flat dab of colour and
  one ink stroke, drawn by `M.icon(name)` in `docs/assets/micro.js`. The dab
  says whose it is or what state it is in: ultramarine for the furniture of a
  lesson, green and raspberry for a right and a wrong guess, a part's own
  pigment for that part. The stroke says what it is. The shared script puts the
  right icon in front of every experiment, "Build it", "Check yourself",
  "What this chapter glossed over", definition, recap, guess and line to keep,
  so a page never draws one itself. A new kind of recurring piece gets one new
  entry in `ICONS`: a `dab(...)` plus one `ink(...)` path on a 48 by 48 grid.
- **Each part has a sign.** The eight parts are listed in `M.parts` with a
  pigment each. A part's sign is its pigment as the dab and a gesture derived
  from its name, shown in the contents and at the top of each of its chapters.
  The eight pigments are used for nothing else, and none is close to the three
  meaning colours.
- **Four more pigments say "which one".** `--pig1` to `--pig4` tell apart
  several things of the same kind in one experiment, such as the four experts
  of chapter 20. They never encode a value, and each use also carries a number
  or a label, so colour alone never carries the meaning.
- **Controls are small and quiet.** `M.button(label, fn)` is an underlined
  word; pass `'go'` for the one main action of an experiment. No filled
  buttons. A tap target is at least 44 px tall.
- **Phone first.** The prose column is 640 px at most, the base size 18 px.
  Check every change at a width of 390 px, in light and in dark, and make sure
  the page does not scroll sideways.

## Checking your change

```bash
python3 tools/check_site.py          # links, glossary keys, JS/Python parity,
                                     # and every page in headless Chromium
python3 tools/check_site.py --fast   # the same without the browser pass
```

Both must pass before a pull request is merged.

## Pull requests

Work on a branch named `type/short-slug` (`docs/`, `feat/`, `fix/`, `test/`)
and open a pull request with a conventional-commit title. Do not push to
`main`. One topic per pull request.
