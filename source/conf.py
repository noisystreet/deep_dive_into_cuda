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

# 全局 Mermaid 渲染：放大字体、放宽横向压缩，避免复杂图被挤太小
mermaid_width = '100%'
mermaid_height = 'auto'
mermaid_init_config = {
    'startOnLoad': False,
    'theme': 'default',
    'themeVariables': {
        'fontSize': '18px',
        'fontFamily': '"Noto Sans SC", "Source Han Sans SC", "PingFang SC", sans-serif',
    },
    'flowchart': {
        'useMaxWidth': False,
        'htmlLabels': True,
        'nodeSpacing': 50,
        'rankSpacing': 55,
        'padding': 12,
    },
    'sequence': {
        'useMaxWidth': False,
        'messageFontSize': '16px',
        'noteFontSize': '16px',
        'actorFontSize': '16px',
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
