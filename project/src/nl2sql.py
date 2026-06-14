"""NL2SQL Engine — converts natural language questions to SQL queries."""
from __future__ import annotations

from .llm_client import LLMClient

NL2SQL_SYSTEM_PROMPT = """你是一个SQL专家。根据给定的数据库Schema，将用户的自然语言问题转换为SQLite SQL查询。

规则：
1. 只返回SQL语句，不要有任何解释
2. 使用SQLite兼容的语法
3. 不要使用MySQL/PostgreSQL特有的函数
4. 列名和表名不需要加引号，除非包含特殊字符
5. 返回格式必须是纯SQL，不要用markdown代码块包裹
6. 如果问题无法转换为SQL，返回: NOT_APPLICABLE"""

NL2SQL_USER_TEMPLATE = """数据库Schema:
{schema}

问题: {question}

SQL:"""


class NL2SQLEngine:
    """Converts NL questions to SQL using an LLM."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def generate_sql(self, question: str, schema_text: str) -> str:
        """Generate SQL from a natural language question and database schema."""
        messages = [
            {"role": "system", "content": NL2SQL_SYSTEM_PROMPT},
            {"role": "user", "content": NL2SQL_USER_TEMPLATE.format(
                schema=schema_text,
                question=question,
            )},
        ]
        raw = self.llm.chat(messages, temperature=0.0)
        return self._clean_sql(raw)

    def _clean_sql(self, text: str) -> str:
        """Extract clean SQL from LLM output."""
        text = text.strip()
        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove opening fence
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        # Remove trailing semicolons
        text = text.rstrip(";").strip()
        return text
