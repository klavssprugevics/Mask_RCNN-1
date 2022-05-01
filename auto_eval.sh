declare -a models=(
"roof20220421T0841"
"roof20220421T1124"
"roof20220421T1712"
"roof20220422T0706"
"roof20220422T0924"
"roof20220422T1151"
)

for i in "${models[@]}"
do
   python roof_plane_segmentation.py predict --weights=./models_new/"$i"/mask_rcnn_roof_0120.h5 --resultout ./results/"$i"/

   echo "$i"
   python roof_plane_segmentation.py eval --weights=./models_new/"$i"/mask_rcnn_roof_0120.h5 --dataset=./RoofPlaneDataset2/large_test/cir
   echo "--------------------------------------------"
done

