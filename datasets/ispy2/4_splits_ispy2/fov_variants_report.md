# ispy2 FOV-variant derivation — verification report

Patients processed: **560** (438 natively unilateral, 122 natively bilateral); failures: 0

## Case counts per contrast per FOV

| contrast | bilateral | unilateral | total |
|---|---|---|---|
| t1wce | 122 | 560 | 682 |
| t2w | 560 | 560 | 1120 |

Derived-side selection (bilateral patients, lesion-centroid half): {'right': 57, 'left': 65}

## Orientation

Volumes with axcodes != LPS: **0** (all native and all derived images and masks are LPS)

## Lesion survival through the crop

Crops that LOST the lesion entirely: **0** — none.

Crops that partially clipped the lesion: **5**

- sub-ispy2199960 / t1wce (survival 0.9948)
- sub-ispy2255078 / t1wce (survival 0.9990)
- sub-ispy2411806 / t1wce (survival 0.9810)
- sub-ispy2602943 / t1wce (survival 0.9957)
- sub-ispy2676365 / t1wce (survival 0.9994)

## All warnings (60 patients)

- `sub-ispy2117707` (uni): T1 in-plane extent only [0.962, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2153327` (uni): T1 in-plane extent only [0.9716, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2155171` (uni): T1 in-plane extent only [0.9695, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2194024` (uni): T1 in-plane extent only [0.9547, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2196024` (uni): T1 in-plane extent only [0.966, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2199960` (bil): axis-1 window shifted by -3 voxels to contain the lesion; T1wce crop clipped the lesion (survival 0.9948)
- `sub-ispy2213328` (uni): T1 in-plane extent only [0.9653, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2228340` (uni): T1 in-plane extent only [0.9669, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2245655` (uni): T1 in-plane extent only [1.0, 0.8412] contained in T2 grid (crop clipped)
- `sub-ispy2255078` (bil): T1wce crop clipped the lesion (survival 0.999)
- `sub-ispy2255535` (uni): T1 in-plane extent only [0.9706, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2262131` (uni): T1 in-plane extent only [0.975, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2266509` (uni): T1 in-plane extent only [0.9756, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2284355` (uni): T1 in-plane extent only [0.9285, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2287300` (uni): T1 in-plane extent only [0.9716, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2294265` (uni): T1 in-plane extent only [0.9594, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2345574` (uni): T1 in-plane extent only [0.9791, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2363669` (uni): T1 in-plane extent only [0.9377, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2373672` (uni): T1 in-plane extent only [1.0, 0.979] contained in T2 grid (crop clipped)
- `sub-ispy2411806` (bil): axis-1 window shifted by 24 voxels to contain the lesion; T1wce crop clipped the lesion (survival 0.981)
- `sub-ispy2421829` (bil): axis-1 window shifted by 20 voxels to contain the lesion
- `sub-ispy2423979` (uni): T1 in-plane extent only [0.9741, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2443444` (bil): axis-1 window shifted by -6 voxels to contain the lesion
- `sub-ispy2453596` (uni): T1 in-plane extent only [0.8925, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2483910` (bil): axis-1 window shifted by 5 voxels to contain the lesion
- `sub-ispy2484867` (uni): T1 in-plane extent only [0.9638, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2485637` (bil): axis-1 window shifted by 60 voxels to contain the lesion
- `sub-ispy2492612` (uni): T1 in-plane extent only [0.963, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2498985` (bil): axis-1 window shifted by 8 voxels to contain the lesion
- `sub-ispy2528479` (uni): T1 in-plane extent only [0.9149, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2539291` (uni): T1 in-plane extent only [0.9561, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2540539` (bil): axis-1 window shifted by 4 voxels to contain the lesion
- `sub-ispy2570148` (uni): T1 in-plane extent only [0.9048, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2572235` (uni): T1 in-plane extent only [0.9793, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2575457` (uni): T1 in-plane extent only [0.9169, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2590299` (bil): axis-1 window shifted by -1 voxels to contain the lesion
- `sub-ispy2596079` (bil): axis-1 window shifted by 2 voxels to contain the lesion
- `sub-ispy2602943` (bil): axis-1 window shifted by -1 voxels to contain the lesion; T1wce crop clipped the lesion (survival 0.9957)
- `sub-ispy2604326` (uni): T1 in-plane extent only [0.9617, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2627118` (uni): T1 in-plane extent only [0.965, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2630758` (uni): T1 in-plane extent only [0.9742, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2636424` (uni): T1 in-plane extent only [0.8609, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2639663` (bil): axis-1 window shifted by 23 voxels to contain the lesion
- `sub-ispy2640043` (bil): axis-1 window shifted by 50 voxels to contain the lesion
- `sub-ispy2651401` (uni): T1 in-plane extent only [0.9717, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2655795` (bil): axis-1 window shifted by 48 voxels to contain the lesion
- `sub-ispy2676365` (bil): axis-1 window shifted by 18 voxels to contain the lesion; T1wce crop clipped the lesion (survival 0.9994)
- `sub-ispy2677915` (uni): T1 in-plane extent only [0.9613, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2683173` (uni): T1 in-plane extent only [0.9218, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2746265` (uni): T1 in-plane extent only [0.9516, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2759418` (uni): T1 in-plane extent only [0.9542, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2781335` (uni): T1 in-plane extent only [0.9653, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2802105` (uni): T1 in-plane extent only [0.9738, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2832056` (bil): axis-1 window shifted by 57 voxels to contain the lesion
- `sub-ispy2851180` (uni): T1 in-plane extent only [1.0, 0.8899] contained in T2 grid (crop clipped)
- `sub-ispy2883696` (uni): T1 in-plane extent only [0.9754, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2883886` (bil): axis-1 window shifted by -2 voxels to contain the lesion
- `sub-ispy2896939` (bil): axis-1 window shifted by 64 voxels to contain the lesion
- `sub-ispy2916645` (uni): T1 in-plane extent only [0.9749, 1.0] contained in T2 grid (crop clipped)
- `sub-ispy2994987` (uni): T1 in-plane extent only [0.9383, 1.0] contained in T2 grid (crop clipped)
