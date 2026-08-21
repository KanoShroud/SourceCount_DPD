function cfg = gate0_runtime(chapter_name, entry_name)
%GATE0_RUNTIME 项目级 MATLAB 运行模式、数据目录与输出隔离配置。
%
% 默认采用 smoke，避免直接启动正式大规模数据生成。正式运行前需显式执行：
%   setenv('SOURCECOUNT_RUN_MODE', 'formal')
%
% 可选环境变量：
%   SOURCECOUNT_DATA_ROOT   正式数据根目录，默认 <project>/data
%   SOURCECOUNT_OUTPUT_ROOT 输出根目录，默认 <project>/outputs

project_root = fileparts(mfilename('fullpath'));
run_mode = lower(strtrim(getenv('SOURCECOUNT_RUN_MODE')));
if isempty(run_mode)
    run_mode = 'smoke';
end
if ~ismember(run_mode, {'smoke', 'formal'})
    error('SOURCECOUNT_RUN_MODE 必须为 smoke 或 formal，当前为 %s', run_mode);
end

data_root = strtrim(getenv('SOURCECOUNT_DATA_ROOT'));
if isempty(data_root)
    data_root = fullfile(project_root, 'data');
end
output_root = strtrim(getenv('SOURCECOUNT_OUTPUT_ROOT'));
if isempty(output_root)
    output_root = fullfile(project_root, 'outputs');
end

if strcmp(run_mode, 'smoke')
    data_dir = fullfile(output_root, 'smoke', chapter_name, 'data');
else
    data_dir = fullfile(data_root, chapter_name);
end
entry_output_dir = fullfile(output_root, run_mode, chapter_name, entry_name);

if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(entry_output_dir, 'dir'), mkdir(entry_output_dir); end

cfg = struct();
cfg.project_root = project_root;
cfg.mode = run_mode;
cfg.is_smoke = strcmp(run_mode, 'smoke');
cfg.data_dir = data_dir;
cfg.output_dir = entry_output_dir;

fprintf('[Gate0] mode=%s\n', cfg.mode);
fprintf('[Gate0] data=%s\n', cfg.data_dir);
fprintf('[Gate0] output=%s\n', cfg.output_dir);
if cfg.is_smoke
    fprintf('[Gate0] 默认 smoke 已启用；不会写入正式数据目录。\n');
end
end
