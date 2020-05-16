import math
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=int, help="input dimension (assume h==w)", required=True)
    parser.add_argument("--k", type=int, help="filter size (assume h==w)", default=3)
    parser.add_argument("--s-range",type=int, help="stride range", default=1)
    parser.add_argument("--p-range",type=int, help="padding range",default=4)
    return parser.parse_args()
    
def main(args):
    args = parse_args()
    dilation = 1
    stride_range = args.s_range
    pad_range = args.p_range
    in_dim = args.input
    filter_size = args.k
    
    possibilites = 0

    for padding in range(pad_range+1):
        for stride in range(1,stride_range+1):
            out_dim = (in_dim + 2*padding - filter_size) / (stride) + 1
            if(out_dim - out_dim//1 == 0):
                print("In: {}, k: {}, s: {}, p:{}, Out:{}".format(in_dim, filter_size, stride, padding, int(out_dim)))
                possibilites += 1

    if (not possibilites):
        print("No integral output dimension possible with the given configuration")

if __name__=="__main__":
    main(sys.argv)


