import Image
inf = float('inf')

def makeinf(vals):
    return map(lambda sequence : map(lambda x: x if x > 0 else inf, sequence), vals)

if __name__ == '__main__':
    filename = "P1/G6/1"
    vals = []
    for line in open("Depth/"+filename+'.txt','r').readlines():
        vals.append(map(int,line.strip().split(',')))
    vals = makeinf(vals)

    minimum = min(map(min,vals))
    print minimum
    for line in vals:
            try:
                print line.index(minimum), vals.index(line)
            except ValueError:
                continue

    rgb = Image.open("Image/"+filename+'.jpg')
    rgb.show()
    depth = Image.new('L',(640,480))
    flat = []
    def to_pixel(x):
        #if x == inf:
        #    x = 16384
        return min((x-minimum)*4,255)
    map(flat.extend,vals)
    
    depth.putdata(map(to_pixel,flat))
    depth.show()
    print "%d colors" % len(depth.getcolors())
