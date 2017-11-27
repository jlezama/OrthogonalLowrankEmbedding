% example script for visualizing deep features
load results/wd_0.001_batch_32_channel_32_samples_500/3cde8d55-42e5-4213-8ac8-8bf4975e8d85/00163.mat
[V I] = sort(labels_test);
d = pdist2(features_test(I,:),features_test(I,:),'cosine');
hfig = figure();
d2 = acos(1-d)*180/pi;
d2 = min(d2, 180-d2);
imagesc(d2, [0 90]);
axis image;colorbar;
