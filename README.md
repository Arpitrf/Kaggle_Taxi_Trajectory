***Tips***
  
1. To use matplotlib in virtualenv (mac OSX) paste following in ~/.bash_profile:

```
function frameworkpython {
    if [[ ! -z "$VIRTUAL_ENV" ]]; then
        PYTHONHOME=$VIRTUAL_ENV /usr/local/bin/python3 "$@"
    else
        /usr/local/bin/python3 "$@"
    fi
}
```

2. TensorFlow:
  1. Python 3.7 does not work.
  2. brew unlink python
  3. Installs Python 3.6.5 - brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb

3. Do not create virtualenv behind proxy. (Virtualenv won't be created)
