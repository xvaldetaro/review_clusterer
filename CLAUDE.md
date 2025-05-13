## General Python project guidelines
(I will already have the setup done by myself. No need to do anything, but just know that this is it)
1. Python 3.12 using pyenv local
2. Use Poetry for dependency management and packaging
3. Run code with `poetry run <command>`

## Avoid unnecessary state
Avoid having unnecessary Class members in python. I commonly see generated files having a generated class that receives a few parameters in its constructor to simply use them in a function call. Something like:

```
foo = Foo(arg1, arg2)
foo.bar()
```

instead of simply

```
foo = bar(arg1, arg2)
# or
foo = Foo()
foo.bar(arg1, arg2)
```

Only keep local state if necessary, e.g. if you create a DB wrapper for a certain file, then keep that file as state. Otherwise just use pure functions and use Classes as just a way to group functions.

## Avoid unnecessary **init**.py files

Only create **init**.py files in the root package to make it importable. Don't add empty **init**.py files in subfolders unless they are specifically needed for imports.
