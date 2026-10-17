# The README's example figures

Real data only. Run with a Python that has numpy, matplotlib, scipy, rasterio, pystac-client and planetary-computer;
the imagery comes from Microsoft Planetary Computer (no key) into a folder of your choice.

**Peru (Sentinel-2).** `oe-inferencex demo` in a folder `DIR/demo`, then
`python fetch_peru.py DIR/data_peru` (Sentinel-2 L2A of 9 June 2019, the Dynamic World tile's date, 1.5 MB) and
`python fig_peru.py DIR example_peru.png`.

**Lake Shkodër (Sentinel-1 and Sentinel-2).** `python locate_flood.py CHIP.npz DIR/data_flood` finds the comparison
figure's 64 x 64 chip on the 28 March 2018 scene by matching its Sentinel-2 water index (`CHIP.npz` is
`docs/figures/tikz/rasters/cmp_square7.npz` in the research record), then `python fetch_flood_context.py DIR/data_flood`
and `python fetch_flood_context.py DIR/data_flood s1only` (8 km around it, about 8 MB) and
`python fig_flood.py DIR/data_flood example_flood_context.png`.

Imagery: contains modified Copernicus Sentinel data (2017 to 2019). Map and expert labels: Dynamic World test tiles
(Brown et al., Scientific Data, 2022), CC BY 4.0.
