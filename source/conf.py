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

# 全局 Mermaid 渲染：与 sphinx_rtd_theme 协调的尺寸
mermaid_width = '100%'
mermaid_height = 'auto'
mermaid_init_config = {
    'startOnLoad': False,
    'theme': 'neutral',
    'themeVariables': {
        'fontSize': '14px',
        'primaryTextColor': '#404040',
        'secondaryColor': '#f5f5f5',
        'tertiaryColor': '#fff',
        'lineColor': '#666',
        'fontFamily': '"Lato", "Noto Sans SC", "Source Han Sans SC", "PingFang SC", sans-serif',
    },
    'flowchart': {
        'useMaxWidth': True,
        'htmlLabels': True,
        'nodeSpacing': 30,
        'rankSpacing': 35,
        'padding': 6,
    },
    'sequence': {
        'useMaxWidth': True,
        'messageFontSize': '13px',
        'noteFontSize': '13px',
        'actorFontSize': '13px',
    },
}

templates_path = ['_templates']
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
