declare -a models=(
"roof20220420T0441"
"roof20220420T0759"
"roof20220420T1209"
"roof20220420T1455"
"roof20220420T1731"
"roof20220421T0534"
)

for i in "${models[@]}"
do
   python roof_plane_segmentation_4dim.py predict --weights=./models_new/"$i"/mask_rcnn_roof_0120.h5 --resultout ./results/"$i"/

   echo "$i"
   python roof_plane_segmentation_4dim.py eval --weights=./models_new/"$i"/mask_rcnn_roof_0120.h5 --dataset=./RoofPlaneDataset2/large_test/cir_ndsm
   echo "--------------------------------------------"
done

