***Tips***
  
1. Tou use matplotlib in virtualenv (mac OSX) paste following in ~/.bash_profile:

`function frameworkpython {
    if [[ ! -z "$VIRTUAL_ENV" ]]; then
        PYTHONHOME=$VIRTUAL_ENV /usr/local/bin/python3 "$@"
    else
        /usr/local/bin/python3 "$@"
    fi
}`

