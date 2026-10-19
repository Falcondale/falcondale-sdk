# Security Policy

## Supported versions

Only the latest released version of Falcondale SDK receives security fixes.

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Use GitHub's [private vulnerability reporting](https://github.com/Falcondale/falcondale-sdk/security/advisories/new) instead, or write to [info@falcondale.pro](mailto:info@falcondale.pro). Include as much information as you can: affected versions, steps to reproduce and the potential impact. We will acknowledge your report as soon as possible and keep you informed while we work on a fix.

## Good to know

Falcondale models are stored using Python's `pickle` module (`Model.save` and `Model.load`). Unpickling data can execute arbitrary code, so only load model files coming from sources you trust.
