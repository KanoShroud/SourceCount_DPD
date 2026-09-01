function cfg = gate0_runtime(chapter_name, entry_name)
%GATE0_RUNTIME 项目级 MATLAB 运行模式、数据目录与输出隔离配置。
%
% 默认采用 smoke，避免直接启动正式大规模数据生成。正式运行前需显式执行：
%   setenv('SOURCECOUNT_RUN_MODE', 'formal')
%
% 可选环境变量：
%   SOURCECOUNT_DATA_ROOT   正式数据根目录，默认 <project>/data
%   SOURCECOUNT_OUTPUT_ROOT           输出根目录，默认 <project>/outputs_e2e
%   SOURCECOUNT_REFERENCE_OUTPUT_ROOT 原项目冻结outputs根目录，只读

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
    output_root = fullfile(project_root, 'outputs_e2e');
end
reference_output_root = strtrim(getenv('SOURCECOUNT_REFERENCE_OUTPUT_ROOT'));
if ~isempty(reference_output_root)
    reference_output_root = canonical_path(reference_output_root);
    output_root = canonical_path(output_root);
    data_root = canonical_path(data_root);
    if ~isfolder(reference_output_root)
        error('只读参考根不存在: %s', reference_output_root);
    end
    reference_prefix = [lower(reference_output_root), filesep];
    output_prefix = [lower(output_root), filesep];
    if strcmpi(reference_output_root, output_root) || ...
            startsWith(lower(output_root), reference_prefix) || ...
            startsWith(lower(reference_output_root), output_prefix)
        error('输出根与只读参考根必须不同且互不嵌套。');
    end
    if strcmpi(reference_output_root, data_root) || ...
            startsWith(lower(data_root), reference_prefix)
        error('数据写入根不得位于原项目冻结outputs内。');
    end
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
cfg.reference_output_root = reference_output_root;

fprintf('[Gate0] mode=%s\n', cfg.mode);
fprintf('[Gate0] data=%s\n', cfg.data_dir);
fprintf('[Gate0] output=%s\n', cfg.output_dir);
if ~isempty(cfg.reference_output_root)
    fprintf('[Gate0] reference_outputs=%s [READ ONLY]\n', ...
        cfg.reference_output_root);
end
if cfg.is_smoke
    fprintf('[Gate0] 默认 smoke 已启用；不会写入正式数据目录。\n');
end
end

function path_out = canonical_path(path_in)
path_out = char(java.io.File(path_in).getCanonicalPath());
end
