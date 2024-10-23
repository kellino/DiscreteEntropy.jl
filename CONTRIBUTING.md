# Contributing

Please contribute! Contributions are very welcome and greatly appreciated.
Credit will always be given, and every little helps.

## Bug reports

When reporting a [bug](https://github.com/kellino/DiscreteEntropy.jl/issues) please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

## Documentation improvements

`DiscreteEntropy.jl` could always use more documentation, whether as part of the
official docs, in docstrings, or even on the web in blog posts, articles, and such.

## Entropy Estimators

There are always new estimators being proposed: `DiscreteEntropy.jl` has a lot but is far from exhaustive.
All suggestions for new (discrete) estimators are welcome, even if they are (conditional) mutual information
estimators or KL estimators, rather than just entropy estimators.

If you are proposing a new estimator:

- Please provide a link to the original paper/blog
- The authors' original code, if available, is extremely useful. In additional to
  providing implementation suggestions, it can also be used to provide ground truth results for testing
- An implemention in Julia would be a wonderful contribution

## Feature requests and feedback

The best way to send feedback is to file an issue [here](https://github.com/kellino/DiscreteEntropy.jl/issues)

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Parts of `DiscreteEntropy.jl` are quite experimental and not based on any existing paper.
Evaluation and feedback on what works and what does not is always interesting and valuable.

## Development

To set up `DiscreteEntropy.jl` for local development:

1. Fork [`DiscreteEntropy.jl`](https://github.com/kellino/DiscreteEntropy.jl)

2. Clone your fork locally:

   ```
   git clone git@github.com:<YOUR ACCOUNT NAME>/DiscreteEntropy.jl.git
   ```

3. Create a branch for local development::

   git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally

4. Set yourself up for local [development](https://julialang.org/contribute/developing_package/)

5. When you're done making changes run all the tests for the project root directory:

   ```
    julia> ]
    pkg> dev .
    pkg> activate .
    pkg> test
   ```

6. Commit your changes and push your branch to GitHub:

```
   git add .
   git commit -m "detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run `test` in the Julia repl).
2. Update documentation when there's new API, functionality etc.
3. Add a note to `CHANGELOG.md` about the changes.
4. Add yourself to `AUTHORS.md`.
