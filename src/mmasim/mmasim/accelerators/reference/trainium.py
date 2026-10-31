"""Trainium3 arithmetic candidate; validation scope in README.rst.
Values are signed integer significands times powers of two. Intermediate
significands are rounded without assuming an architectural exponent width.
"""
from .arithmetic import unpack,shift_round,pack32,add32

def quant(value,p=24,mode='rne'):
 n,e=value
 if not n:return (0,0)
 shift=max(0,abs(n).bit_length()-p)
 return shift_round(n,-shift,mode),e+shift

def add(a,b,p=24):
 an,ae=a;bn,be=b
 if not an:return quant(b,p)
 if not bn:return quant(a,p)
 e=min(ae,be);return quant(((an<<(ae-e))+(bn<<(be-e)),e),p)

def reduce_terms(terms,chain=32):
 partials=[]
 for i in range(0,len(terms),chain):
  acc=(0,0)
  for term in terms[i:i+chain]:acc=add(acc,term)
  partials.append(acc)
 while len(partials)>1:partials=[add(partials[i],partials[i+1]) if i+1<len(partials) else partials[i] for i in range(0,len(partials),2)]
 return partials[0] if partials else (0,0)

def scalar_products(a,b):
 terms=[]
 for x,y in zip(a,b):
  an,ae=unpack(x,23,127,ftz=False);bn,be=unpack(y,23,127,ftz=False)
  terms.append((an*bn,ae+be))
 return terms

def ordinary(a,b,chunk=128):
 terms=scalar_products(a,b);acc=0
 for i in range(0,len(terms),chunk):
  value=pack32(*reduce_terms(terms[i:i+chunk]));acc=add32(acc,value,ftz=False)
 return acc

def sum_exact(values):
 values=[(n,e) for n,e in values if n]
 if not values:return (0,0)
 e=min(e for n,e in values);return sum(n<<(q-e) for n,q in values),e

def fraction_term(value):
 denominator=value.denominator
 assert denominator&(denominator-1)==0
 return value.numerator,-(denominator.bit_length()-1)

def extension(row,kind,mapping='half',route='single',parts=128,mx_chain=16,
              pair_precision=11,partition_precision=11,mx_order=(0,2,1,3)):
 from fractions import Fraction as F
 scales={int(k):v for k,v in row.get('scales_unbiased',{}).items()}
 terms=[(0,0)]*row['logical_k']
 for t,x,y in row['terms']:
  n,e=fraction_term(F(x)*F(y));terms[t]=(n,e+sum(scales.get(t//32,[0,0])))
 if kind=='double_row':
  if route=='two_inst':
   acc=0
   for half in range(2):
    values=terms[half*128:(half+1)*128] if mapping=='half' else terms[half::2]
    acc=add32(acc,pack32(*reduce_terms(values,32)),ftz=False)
   return acc
  pairs=[(terms[i],terms[i+128]) for i in range(128)] if mapping=='half' else [terms[i:i+2] for i in range(0,256,2)]
  values=[quant(sum_exact(pair),pair_precision) for pair in pairs]
  return pack32(*reduce_terms(values,32))
 # Packed x4: two p=11 pair sums, followed by a p=11 partition sum.
 pairs=[quant(sum_exact(terms[i:i+2]),pair_precision) for i in range(0,len(terms),2)]
 values=[quant(sum_exact(pairs[i:i+2]),partition_precision) for i in range(0,len(pairs),2)]
 acc=0
 for i in range(0,len(values),parts):
  tile=values[i:i+parts]
  ordered=[tile[b+j*8+t] for b in range(0,len(tile),32) for j in mx_order for t in range(8) if b+j*8+t<len(tile)]
  value=pack32(*reduce_terms(ordered,mx_chain));acc=add32(acc,value,ftz=False)
 return acc
