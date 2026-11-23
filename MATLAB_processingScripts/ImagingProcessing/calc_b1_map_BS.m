function [B1_norm, R2_vol] = calc_b1_map_BS(imgData, pwBS, BSoffset, mask)
%% calc_b1_map_BS  –  B1 map from Bloch-Siegert shift imaging data
%
% USAGE
%   [B1_norm, R2_vol] = calc_b1_map_BS(imgData, pwBS, BSoffset)
%   [B1_norm, R2_vol] = calc_b1_map_BS(imgData, pwBS, BSoffset, mask)
%
% INPUTS
%   imgData  - 4D complex array [Nx x Ny x Nz x N_pw]
%              4th dimension corresponds to BS pulse widths
%   pwBS     - [1 x N_pw] or [N_pw x 1] BS pulse widths in seconds
%   BSoffset - BS off-resonance frequency in Hz (scalar)
%   mask     - (optional) 3D logical array [Nx x Ny x Nz].
%              If omitted, voxels with mean SNR > 3 are used.
%
% OUTPUTS
%   B1_norm  - 3D normalised B1 amplitude map [0, 1]  (NaN outside mask)
%   R2_vol   - 3D linear-fit R² map             [0, 1]  (NaN outside mask)
%
% METHOD
%   1. Build mask (SNR > 3) if not provided
%   2. Sort pwBS ascending; reorder imgData 4th dim to match
%   3. Extract phase; unwrap along pulse-width dimension per voxel
%   4. Fit phase vs. pulse width to a line (vectorised); slope = w_BS [rad/s]
%   5. Convert to B1 nutation frequency:
%        phi_BS = (w1^2 / 2*Delta_omega) * pw_BS
%        => w_BS_Hz = w1_Hz^2 / (2 * f_BS)
%        => w1_Hz   = sqrt( w_BS_Hz * 2 * f_BS )
%   6. Normalise by peak in-mask value
%
% Written 2026-06-01

%% ---- 1. Parse and validate inputs ---------------------------------------
narginchk(3, 4);

[Nx, Ny, Nz, N_pw] = size(imgData);
pwBS  = double(pwBS(:));       % column vector [N_pw x 1], seconds
f_BS  = double(BSoffset);      % BS offset frequency, Hz

assert(numel(pwBS) == N_pw, ...
    'calc_b1_map_BS:dimMismatch', ...
    'Length of pwBS (%d) must match the 4th dimension of imgData (%d).', ...
    numel(pwBS), N_pw);
assert(isscalar(f_BS) && f_BS > 0, ...
    'calc_b1_map_BS:invalidOffset', ...
    'BSoffset must be a positive scalar (Hz).');

fprintf('imgData : [%d x %d x %d x %d] complex\n', Nx, Ny, Nz, N_pw);
fprintf('pwBS    : %s s\n', mat2str(pwBS', 4));
fprintf('BSoffset: %.1f Hz\n', f_BS);

%% ---- 2. Sort pwBS ascending; reorder imgData to match -----------------
[pwBS, sort_idx] = sort(pwBS, 'ascend');
imgData = imgData(:, :, :, sort_idx);

%% ---- 3. Build mask (SNR > 3) or validate user-supplied mask ------------
mag      = abs(imgData);            % [Nx x Ny x Nz x N_pw]
mean_mag = mean(mag, 4);

if nargin < 4 || isempty(mask)
    % Noise estimate: std of magnitude in the four spatial corners of the
    % first BS sub-image, pooled across all slices.
    csz = 8;
    bg_voxels = [
        reshape(mag(1:csz,         1:csz,         :, 1), [], 1);
        reshape(mag(end-csz+1:end, 1:csz,         :, 1), [], 1);
        reshape(mag(1:csz,         end-csz+1:end, :, 1), [], 1);
        reshape(mag(end-csz+1:end, end-csz+1:end, :, 1), [], 1)
    ];
    noise_std = std(bg_voxels);

    if noise_std == 0
        warning('calc_b1_map_BS:zeroNoise', ...
            'Corner noise estimate is zero; using 1%% of mean signal as fallback.');
        noise_std = 0.01 * mean(mean_mag(:), 'omitnan');
    end

    SNR_thresh = 3;
    SNR_map    = mean_mag / noise_std;
    mask       = SNR_map > SNR_thresh;

    fprintf('SNR mask (>%d): %d / %d voxels (%.1f%%)\n', ...
        SNR_thresh, sum(mask(:)), numel(mask), 100 * mean(mask(:)));
else
    assert(isequal(size(mask), [Nx, Ny, Nz]) && islogical(mask), ...
        'calc_b1_map_BS:invalidMask', ...
        'mask must be a logical array of size [%d x %d x %d].', Nx, Ny, Nz);
    SNR_map = [];
    fprintf('User-supplied mask: %d / %d voxels (%.1f%%)\n', ...
        sum(mask(:)), numel(mask), 100 * mean(mask(:)));
end

%% ---- 4. Phase extraction and per-voxel unwrapping ----------------------
% angle() returns values in (-pi, pi]. unwrap() along dim 4 adjusts each
% consecutive step to be within [-pi, pi], correcting 2*pi jumps.
phi    = angle(imgData);        % [Nx x Ny x Nz x N_pw]
phi_uw = unwrap(phi, [], 4);    % unwrap along BS pulse-width dimension

%% ---- 5. Vectorised voxelwise linear fit --------------------------------
% Reshape to [N_pw x N_vox] so one matrix solve handles all voxels.
N_vox  = Nx * Ny * Nz;
phi_2d = reshape(permute(phi_uw, [4, 1, 2, 3]), N_pw, N_vox);  % [N_pw x N_vox]

% Design matrix: phi = slope * pwBS + intercept
X      = [pwBS, ones(N_pw, 1)];         % [N_pw x 2]
coeffs = (X' * X) \ (X' * phi_2d);     % [2 x N_vox]; row 1 = slope [rad/s]

% Bloch-Siegert shift (rad/s), reshaped to 3D
w_BS_vol = reshape(coeffs(1, :), Nx, Ny, Nz);

% R² per voxel
y_hat  = X * coeffs;                               % [N_pw x N_vox], predicted phase
SS_res = sum((phi_2d - y_hat).^2, 1);              % residual sum of squares
SS_tot = sum((phi_2d - mean(phi_2d, 1)).^2, 1);    % total sum of squares
R2_vol = reshape(1 - SS_res ./ SS_tot, Nx, Ny, Nz);

%% ---- 6. Convert BS shift to B1 amplitude --------------------------------
% Phase accumulation for a rectangular off-resonance pulse:
%   phi_BS  = (w1^2 / 2*Delta_omega) * pw_BS
%   slope   = w1^2 / (2 * 2*pi * f_BS)          [rad/s]
%   w_BS_Hz = slope / (2*pi) = w1_Hz^2 / (2 * f_BS)
%   => w1_Hz = sqrt( w_BS_Hz * 2 * f_BS )
%
% abs() guards against small negative slopes due to noise before sqrt.
w_BS_Hz = w_BS_vol / (2 * pi);              % [Nx x Ny x Nz], Hz
w1_Hz   = sqrt(abs(w_BS_Hz) * 2 * f_BS);   % nutation frequency, Hz

%% ---- 7. Apply mask and normalise ----------------------------------------
w1_Hz(~mask)  = NaN;
R2_vol(~mask) = NaN;

B1_max  = max(w1_Hz(:), [], 'omitnan');
B1_norm = w1_Hz / B1_max;                  % [0, 1] normalised map

fprintf('Peak B1 (in-mask) : %.2f Hz\n', B1_max);
fprintf('Median R²  (in-mask) : %.4f\n', median(R2_vol(mask), 'omitnan'));

%% ---- 8. Display ----------------------------------------------------------
% Figure sizing: full screen width; height preserves per-slice aspect ratio,
% plus a fixed overhead (px) for title and colorbar label.
scr   = get(0, 'ScreenSize');   % [left bottom width height]
fig_w = scr(3);
deco  = 80;                     % px overhead for title + colorbar label
fig_h = round(fig_w * Nx / (Ny*Nz)) + deco;

% --- Figure 1: Normalised B1, one tile per slice ----------------------
figure('Name', 'Normalised B_1 – Bloch-Siegert', 'NumberTitle', 'off', ...
    'Color', 'w', 'Position', [0, scr(4)-fig_h, fig_w, fig_h]);
tl = tiledlayout(1, Nz, 'TileSpacing', 'none', 'Padding', 'compact');
for sl = 1:Nz
    nexttile;
    imagesc(B1_norm(:, :, sl), [0 1]);
    colormap(gca, 'hot');
    axis image off;
end
cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = 'Normalised B_1';
title(tl, 'Normalised B_1 Map  (Bloch-Siegert)');

% --- Figure 2: Linear-fit R², one tile per slice ----------------------
figure('Name', 'Linear-fit R² – Bloch-Siegert', 'NumberTitle', 'off', ...
    'Color', 'w', 'Position', [0, scr(4)-fig_h, fig_w, fig_h]);
tl = tiledlayout(1, Nz, 'TileSpacing', 'none', 'Padding', 'compact');
for sl = 1:Nz
    nexttile;
    imagesc(R2_vol(:, :, sl), [0.8 1]);
    colormap(gca, 'parula');
    axis image off;
end
cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = 'R²';
title(tl, 'Linear-fit R²  (Bloch-Siegert)');

end
