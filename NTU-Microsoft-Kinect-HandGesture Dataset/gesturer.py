import Image
from numpy import array

def makeinf(vals, infinity):
    return map(lambda sequence : map(lambda x: x if x > 0 else infinity, sequence), vals)

def get_depth_from_vector(image, vector):
    depth = image.getpixel(tuple(map(int,vector)))
    return depth

def extract_feature(image, pixel_x, theta_pxmm):
    u_pxmm, v_pxmm = theta_pxmm
    depth_x_mm = get_depth_from_vector(image, pixel_x)
    return get_depth_from_vector(image, pixel_x+u_pxmm/depth_x_mm) - get_depth_from_vector(image, pixel_x+v_pxmm/depth_x_mm)

if __name__ == '__main__':
    for g in (3,7):
        for p in range(1,11):
            for v in range(1,11):
                filename = "P%d/G%d/%d" % (p,g,v)
                vals = []
                for line in open("Depth/"+filename+'.txt','r').readlines():
                    vals.append(map(int,line.strip().split(',')))


                
                maximum = max(map(max,vals))
                vals = makeinf(vals, 65535)
                minimum = min(map(min,vals))
                #print minimum, maximum
                
                

##                for line in vals:
##                        try:
##                            print line.index(minimum), vals.index(line)
##                        except ValueError:
##                            continue

                rgb = Image.open("Image/"+filename+'.jpg')
                
                depth = Image.new('I',(640,480))
                flat = []
                def extract_foreground(image, minimum, gain):
                    def mapping(x):
                        return min((x-minimum)*4, )
                def foreground_to_24bit(x):
                    #if x == inf:
                    #    x = 16384
                    if x >= 65535:
                        return 128*256
                    gray = (x-minimum)*2
                    if gray > 255:
                        return 128*256*256
                    return gray
                    #return gray
                def normalize_to_8bit(x):
                    return x*255/maximum
                    
                map(flat.extend,vals)
                
                depth.putdata(flat)

                #print extract_feature(depth, array((196,292)), (array((10000,10000)), array((0,0))))
                depth_display = Image.new('RGB',(640,480))
                depth_display.putdata(map(foreground_to_24bit, depth.getdata()))
                #depth_display.save("/tmp/P%dG%dV%d.jpg" % (p,g,v))
##                depth_display.show()
                #depth_norm = Image.new('L',((640,480)))
                #depth_norm.putdata(map(normalize_to_8bit, depth.getdata()))
##                depth_norm.show()
##                Image.blend(rgb,depth_display.convert('RGB'),0.5).show()
                print "%d colors" % len(depth_display.getcolors())
                exit
