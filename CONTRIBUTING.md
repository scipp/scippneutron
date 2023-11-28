## Contributing to ScippNeutron

Welcome to the developer side of ScippNeutron!

Contributions are always welcome.
This includes reporting bugs or other issues, submitting pull requests, requesting new features, etc.

If you need help with using ScippNeutron or contributing to it, have a look at the GitHub [discussions](https://github.com/scipp/scippneutron/discussions) and start a new [Q&A discussion](https://github.com/scipp/scippneutron/discussions/categories/q-a) if you can't find what you are looking for.

For bug reports and other problems, please open an [issue](https://github.com/scipp/scippneutron/issues/new) in GitHub.

You are welcome to submit pull requests at any time.
But to avoid having to make large modifications during review or even have your PR rejected, please first open an issue first to discuss your idea!

Check out the subsections of the [Developer documentation](https://scipp.github.io/scippneutron/developer/index.html) for details on how ScippNeutron is developed.

## Code of conduct

This project is a community effort, and everyone is welcome to contribute.
Everyone within the community is expected to abide by our [code of conduct](https://github.com/scipp/scippneutron/blob/main/CODE_OF_CONDUCT.md).

## Scope

ScippNeutron shall contain only generic neutron-specific functionality.
Facility-specific or instrument-specific functionality must not be added.
Examples of generic functionality that is permitted are

* Unit conversions, which could be generic for all time-of-flight neutron sources.
* Published research such as absorption corrections.

Examples of functionality that shall not be added to ScippNeutron are handling of facility-specific file types or data layouts, or instrument-specific correction algorithms.
The `ess` suite of packages (e.g., [Esssans](https://scipp.github.io/esssans/)) is an example codebase providing facility-specific algorithms
