function results = myimfcn(im)
    % Perform edge detection
    filtered_img = imbilatfilt(im);
    binary_img = imbinarize(filtered_img);
    white_teeth_img = im;
    white_teeth_img(binary_img) = 0;
    grayimg = rgb2gray(white_teeth_img); % Convert the image to grayscale
    eq_img = histeq(grayimg);
    gaussian_img = imgaussfilt(eq_img, 1); 
    canny_edges = edge(gaussian_img, 'sobel');
    
    % Create masks for the top and bottom 1/7 and the left and right 1/8
    [h, w, ~] = size(im);
    top_bottom_mask = false(h, w);
    top_bottom_mask(1:h/7, :) = true;
    top_bottom_mask(end-h/7+1:end, :) = true;
    
    left_right_mask = false(h, w);
    left_right_mask(:, 1:w/8) = true;
    left_right_mask(:, end-w/8+1:end) = true;

    % Set pixel values in masked regions to black
    canny_edges(repmat(top_bottom_mask | left_right_mask, [1, 1])) = 0;

    % Perform morphological closing on the edge image
    se = strel('disk', 15); % Set the size of the structural element for closing
    canny_edges_closed = imclose(canny_edges, se);

    % Fill holes in the edge image
    filled_edges = imfill(canny_edges_closed, 'holes');

    % Compute connected components
    cc = bwconncomp(filled_edges);

    % Retain the largest connected component
    if cc.NumObjects > 1
        numPixels = cellfun(@numel, cc.PixelIdxList);
        [~, idx] = max(numPixels);
        largest_region = false(size(filled_edges));
        largest_region(cc.PixelIdxList{idx}) = true;
        filled_edges = largest_region;
    end

    % Overlay the filled region on the original image
    edges_on_original = im;
    edges_on_original(repmat(filled_edges, [1, 1, 3])) = 0;

    % Save results
    results = edges_on_original;
end

