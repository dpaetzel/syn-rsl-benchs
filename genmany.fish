#!/usr/bin/env fish


# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 1  -K 5   300 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 1  -K 10  300 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 1  -K 20  300 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 3  -K 5   500 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 3  -K 10  500 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 3  -K 20  500 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 5  -K 5  1000 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 5  -K 10 1000 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 5  -K 20 1000 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 10 -K 5  2000 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 10 -K 10 2000 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 10 -K 20 2000 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 20 -K 5  5000 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 20 -K 10 5000 data/rsl
# nix develop --command python gen_data.py genmany --startseed=0 --endseed=9 -d 20 -K 20 5000 data/rsl


nix develop --command \
    parallel \
    python gen_data.py genmany \
    --startseed=0 \
    --endseed=9 \
    -d '{1//}' \
    -K '{2}' '{1/}' \
    data/rsl \
    ::: "1/300" "3/500" "5/1000" "10/2000" "20/5000" \
    ::: 5 10 20
