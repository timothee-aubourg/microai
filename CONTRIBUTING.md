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

Optional maths goes in a folded `<details class="refresher">`. Step-by-step
animations use `M.stepper` (see the BPE walkthrough in chapter 2).

Every number quoted in the text must come from running the code or the data,
not from memory. If a widget's behaviour is described, check it in the browser.

The course does not credit named individuals as inspiration.

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
