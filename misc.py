
import sys

def error(msg):
    print('[-] %s' % (msg,))

def fatal_error(msg):
    print('[-] %s' % (msg,))
    sys.exit(1)

def debug(msg):
    print('[*] %s' % (msg,))

