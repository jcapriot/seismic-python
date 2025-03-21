
# setup spy_config.h
conf = configuration_data()

# determine some sizes
conf.set('SPY_SIZEOF_FLOAT', cc.sizeof('float'))
conf.set('SPY_SIZEOF_DOUBLE', cc.sizeof('double'))

conf.set('HAVE___BUILTIN_BSWAP16', cc.has_function('__builtin_bswap16'))
conf.set('HAVE___BUILTIN_BSWAP32', cc.has_function('__builtin_bswap32'))
conf.set('HAVE___BUILTIN_BSWAP64', cc.has_function('__builtin_bswap64'))

endianness = target_machine.endian()
conf.set('IS_LITTLE_ENDIAN', endianness =='little')
conf.set('IS_BIG_ENDIAN', endianness =='big')

spy_config_h = configure_file(
    input : '_spy_config.h.in',
    output : 'spy_config.h',
    configuration : conf
)

spy_include = include_directories('.')