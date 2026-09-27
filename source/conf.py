# Deep Dive into CUDA - Sphinx Configuration

from datetime import datetime

project = 'Deep Dive Into CUDA'
author = 'deep_dive_into_cuda'
copyright = f'{datetime.now().year}, {author}'

version = '0.1'
release = '0.1'

extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.extlinks',
    'sphinxcontrib.mermaid',
]

mermaid_output_format = 'raw'

# 本项目 Mermaid 图统一使用 neutral 主题的浅色外观。
# sphinxcontrib-mermaid 在渲染时会依据系统/浏览器深色偏好覆盖 theme
# （darkTheme ? mermaid_dark_theme : mermaid_light_theme），
# 因此把两者都固定为 neutral，避免被切换成内置的 dark 主题而出现黑底黑字。
mermaid_dark_theme = 'neutral'
mermaid_light_theme = 'neutral'

# 全局 Mermaid 渲染：与 sphinx_rtd_theme 协调的尺寸
mermaid_width = '100%'
mermaid_height = 'auto'
mermaid_init_config = {
    'startOnLoad': False,
    'theme': 'neutral',
    'themeVariables': {
        'fontSize': '14px',
        # 强制浅色调色板，避免 mermaid 依据 prefers-color-scheme 切换到深色
        'darkMode': False,
        # 文字统一为近黑色
        'primaryTextColor': '#1a1a1a',
        'secondaryTextColor': '#1a1a1a',
        'tertiaryTextColor': '#1a1a1a',
        'textColor': '#1a1a1a',
        # 去掉节点/分组色块：统一白底，仅靠边框与文字区分
        'primaryColor': '#ffffff',
        'secondaryColor': '#ffffff',
        'tertiaryColor': '#ffffff',
        'mainBkg': '#ffffff',
        'clusterBkg': '#ffffff',
        'primaryBorderColor': '#9aa0a6',
        'nodeBorder': '#9aa0a6',
        'clusterBorder': '#c0c0c0',
        'lineColor': '#555',
        'fontFamily': '"Lato", "Noto Sans SC", "Source Han Sans SC", "PingFang SC", sans-serif',
        # 饼图：切片必须靠颜色区分，故例外地保留彩色（Set3 12 色定性调色板），
        # 配深色文字 + 白色描边，浅色切片也不会在白色背景上消失。
        # 注意 mermaid 默认 pie1..pie3 = primaryColor/secondaryColor/tertiaryColor，
        # 它们已被上面改为白色，故这里必须显式给出 pie1..pie12。
        'pie1': '#8DD3C7', 'pie2': '#FFFFB3', 'pie3': '#BEBADA', 'pie4': '#FB8072',
        'pie5': '#80B1D3', 'pie6': '#FDB462', 'pie7': '#B3DE69', 'pie8': '#FCCDE5',
        'pie9': '#D9D9D9', 'pie10': '#BC80BD', 'pie11': '#CCEBC5', 'pie12': '#FFED6F',
        'pieSectionTextColor': '#1a1a1a',
        'pieLegendTextColor': '#1a1a1a',
        'pieTitleTextColor': '#1a1a1a',
        'pieStrokeColor': '#ffffff',
        'pieStrokeWidth': '1px',
        'pieOuterStrokeColor': '#c0c0c0',
        'pieOuterStrokeWidth': '1px',
        'pieOpacity': '1',
    },
    # useMaxWidth=False：让 Mermaid 输出图表的自然像素尺寸，
    # 小图不被拉伸放大；超大图再由 custom.css 的 max-width 限制到容器内自适应缩放。
    'flowchart': {
        'useMaxWidth': False,
        'htmlLabels': True,
        'nodeSpacing': 30,
        'rankSpacing': 35,
        'padding': 6,
    },
    'sequence': {
        'useMaxWidth': False,
        'messageFontSize': '13px',
        'noteFontSize': '13px',
        'actorFontSize': '13px',
    },
    'pie': {
        'useMaxWidth': False,
    },
}

language = 'zh_CN'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
}
html_static_path = ['_static']
html_css_files = ['custom.css']

autosectionlabel_prefix_document = True
todo_include_todos = True
