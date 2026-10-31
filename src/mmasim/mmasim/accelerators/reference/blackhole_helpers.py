from fractions import Fraction
import struct

def f32(x):return struct.unpack('!f',int(x).to_bytes(4,'big'))[0]

def bits(x):return struct.unpack('!I',struct.pack('!f',float(x)))[0]

def p2(e):return Fraction(2)**e

def exponent(x):
 x=abs(x)
 return x.numerator.bit_length()-x.denominator.bit_length()

def rounded(x,rule):
 if rule=='away':
  a=abs(x);q,r=divmod(a.numerator,a.denominator)
  return (q+(2*r>=a.denominator))*(-1 if x<0 else 1)
 q,r=divmod(x.numerator,x.denominator)
 return q+(2*r>=x.denominator)
