import os

domains = ['book','dvd','electronics','kitchen']
for soucre in domains:
    for target in domains:
        cache_path = f'/home/user/gitdir/TLTC/RoBerta_MMD/{soucre}2{target}_cache'
        # os.system(f"bitfusion run -n 1 -- python RoBerta_tl.py --source_domain={soucre} --target_domain={target} --cache_path={cache_path}")
        os.system(f"python /home/user/gitdir/TLTC/RoBerta_MMD/RoBerta_tl.py --source_domain={soucre} --target_domain={target} --cache_path={cache_path}")