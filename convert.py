import re

def convert_latex_to_markdown(text):
    """
    将LaTeX格式的文本转换为指定的Markdown格式
    """
    # 1. 删除 \noindent
    text = text.replace('\\noindent', '')
    
    # 2. 转换 \textbf{...} 为 **...**
    text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', text)
    
    # 3. 转换 \textit{...} 为 *...*
    text = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', text)
    
    # 4. 处理 \cite{...} 引用
    def convert_cite(match):
        citations = match.group(1)
        if ',' in citations:
            # 多个引用
            refs = citations.split(',')
            formatted_refs = ';'.join([f'@{ref.strip()}' for ref in refs])
            return f'[{formatted_refs}]'
        else:
            # 单个引用
            return f'[@{citations.strip()}]'
    
    text = re.sub(r'\\cite\{([^}]+)\}', convert_cite, text)
    
    # 5. 在段落开头添加 &emsp;&emsp;
    lines = text.split('\n')
    processed_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line:
            # 6. 转换 \subsection{...} 为 ## **...**
            subsection_match = re.match(r'\\subsection\{([^}]+)\}', line)
            if subsection_match:
                title = subsection_match.group(1)
                processed_lines.append(f'## **{title}**')
                processed_lines.append('')  # 添加空行
                continue
            
            # 7. 转换 \subsubsection{...} 为 &emsp;&emsp;**) ...**
            subsubsection_match = re.match(r'\\subsubsection\{([^}]+)\}', line)
            if subsubsection_match:
                title = subsubsection_match.group(1)
                processed_lines.append(f'&emsp;&emsp;**) {title}**')
                processed_lines.append('')  # 添加空行
                continue
            
            # 8. 删除 \label{...} 及其内容
            line = re.sub(r'\\label\{[^}]+\}', '', line)
            
            # 9. 转换 \ref{...} 为 **@@@**
            line = re.sub(r'\\ref\{[^}]+\}', '**@@@**', line)
            
            # 10. 将 ~ 转换为空格
            line = line.replace('~', ' ')
            
            # 如果不是标题行，添加段落缩进
            if not line.startswith('##') and not line.startswith('&emsp;&emsp;**)'):
                line = '&emsp;&emsp;' + line
            
            processed_lines.append(line)
        else:
            processed_lines.append('')
    
    return '\n'.join(processed_lines)

def main():
    try:
        # 读取输入文件
        with open('paper.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 转换格式
        converted_content = convert_latex_to_markdown(content)
        
        # 写入输出文件
        with open('result.txt', 'w', encoding='utf-8') as f:
            f.write(converted_content)
        
        print("转换完成！结果已保存到 result.txt")
        
    except FileNotFoundError:
        print("错误：找不到 paper.txt 文件，请确保文件存在于当前目录。")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    main()
