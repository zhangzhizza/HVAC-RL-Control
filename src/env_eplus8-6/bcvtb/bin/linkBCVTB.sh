#!/bin/bash
#echo The BCVTB version is installed in $1.
#echo Remove any existing symbolic link for /usr/local/lib/libbcvtb.dylib
# The symbolic link will be set only if /usr/local/lib exists which
# seems to be true on Mac OS X 10.11 but not on earlier versions of Mac OS X.
if [[ -w /usr/local/lib/ && -w /usr/local/lib/ ]] ; then
  rm -rf /usr/local/lib/libbcvtb.dylib
  #echo Create new symbolic link for /usr/local/lib/libbcvtb.dylib
  ln -s $1"/lib/util/libbcvtb.dylib" "/usr/local/lib/libbcvtb.dylib"
fi
