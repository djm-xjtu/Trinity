import pyopencl as cl

if __name__ == '__main__':
    platforms = cl.get_platforms()
    ctx = cl.Context(dev_type=cl.device_type.ALL,
                     properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(ctx, device=None)
    ml = cl.mem_flags
    
    print(ml)
