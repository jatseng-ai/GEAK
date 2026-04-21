from dataclasses import dataclass
import json
import os
import re
from pathlib import Path

import yaml


@dataclass
class SkillDescriptor:
    name: str
    description: str
    path: Path
    loaded: bool = False  # runtime state


class SkillRuntime:
    """Discovers skills under ``<GEAK repo root>/skills/*/SKILL.md`` and builds prompt / load_skill hooks."""

    def __init__(self):
        repo_root = Path(os.path.dirname(__file__)).resolve().parent.parent.parent
        skills_dir = repo_root / "skills"
        self.skills = self._discover_skills(skills_dir)

    def _extract_yaml_frontmatter(self, markdown: str) -> dict:
        frontmatter_re = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = frontmatter_re.match(markdown)
        if not match:
            raise ValueError("SKILL.md missing YAML frontmatter")

        return yaml.safe_load(match.group(1))

    def _parse_metadata(self, skill_path: Path) -> SkillDescriptor:
        skill_md = skill_path / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")

        fm = self._extract_yaml_frontmatter(content)

        return SkillDescriptor(
            name=fm["name"],
            description=fm["description"],
            path=skill_path,
            loaded=False,
        )

    def _discover_skills(self, skills_root: Path) -> dict[str, SkillDescriptor]:
        if not skills_root.is_dir():
            return {}
        skills: list[SkillDescriptor] = []
        for p in skills_root.iterdir():
            if p.is_dir() and (p / "SKILL.md").exists():
                try:
                    skills.append(self._parse_metadata(p))
                except Exception:
                    print("Get skills fail")
        return {s.name: s for s in skills}

    def build_system_prompt(self) -> str:
        blocks = ["\n<available_skills>"]

        for _, s in self.skills.items():
            blocks.append(
                f"""  <skill>
        <name>{s.name}</name>
        <description>{s.description}</description>
    </skill>"""
            )

        blocks.append("</available_skills>")

        blocks.append(
            """
You can use the above skills.
If a skill is relevant, respond with:

```skills
{
"action": "use_skill",
"skill": "<skill-name>"
}
```
Otherwise, respond normally.
    """
        )

        return "\n".join(blocks)

    @staticmethod
    def _list_skill_material_subdirs(skill_root: Path) -> list[Path]:
        """Immediate subdirectories of the skill folder (same directory as SKILL.md), resolved."""
        root = skill_root.resolve()
        if not root.is_dir():
            return []
        out: list[Path] = []
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir() and not child.name.startswith("."):
                out.append(child.resolve())
        return out

    @staticmethod
    def _format_skill_material_paths(skill_root: Path) -> str:
        """Human-readable block listing material dirs; empty if none."""
        subdirs = SkillRuntime._list_skill_material_subdirs(skill_root)
        if not subdirs:
            return ""
        bullets = "\n".join(f"- `{p}`" for p in subdirs)
        return (
            "\n\n## Skill material paths\n\n"
            "Material for this skill (e.g. docs/, scripts/) is available under these directories:\n\n"
            f"{bullets}\n"
        )

    def load_skill(self, response: dict) -> dict:
        results = {
            "output": "",
            "returncode": 0,
        }
        content = response.get("content")
        if not content:
            return results
        match = re.search(r"```skills\s*(\{.*?\})\s*```", content, re.DOTALL)
        if not match:
            return results
        try:
            payload = json.loads(match.group(1))
            if payload["action"] == "use_skill":
                if payload["skill"] not in self.skills:
                    results["output"] = f"The skill {payload['skill']} is not exist."
                    return results
                skill = self.skills[payload["skill"]]
                if skill.loaded:
                    return results
                skill_md = skill.path / "SKILL.md"
                md_content = skill_md.read_text(encoding="utf-8")
                material = self._format_skill_material_paths(skill.path)
                results["output"] = f"\n# Loaded skill: {skill.name}{material}\n{md_content}"
                skill.loaded = True
        except Exception as e:
            results["output"] = f"No skills. Error: {e}"
        return results
