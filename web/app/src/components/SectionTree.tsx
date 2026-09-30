// The section tree (SPEC §1): clicking a node anchors the section-scoped
// discussion; the anchor is clearable (select_section with "").
import type { SectionNode } from "../types";

interface NodeProps {
  node: SectionNode;
  depth: number;
  selected: string | null;
  onSelect: (path: string) => void;
}

function Node({ node, depth, selected, onSelect }: NodeProps) {
  return (
    <li>
      <button
        className="tree-node"
        style={{ paddingLeft: `${0.5 + depth * 0.9}rem` }}
        aria-current={selected === node.path ? "true" : undefined}
        onClick={() => onSelect(node.path)}
      >
        {node.title ?? node.path}
      </button>
      {node.children.length > 0 && (
        <ul>
          {node.children.map((c) => (
            <Node key={c.path} node={c} depth={depth + 1} selected={selected} onSelect={onSelect} />
          ))}
        </ul>
      )}
    </li>
  );
}

interface Props {
  tree: SectionNode[];
  selected: string | null;
  onSelect: (path: string) => void;
}

export function SectionTree({ tree, selected, onSelect }: Props) {
  return (
    <nav className="tree" aria-label="Sections">
      {selected && (
        <button className="clear-anchor" onClick={() => onSelect("")}>
          Clear anchor ({selected})
        </button>
      )}
      <ul>
        {tree.map((n) => (
          <Node key={n.path} node={n} depth={0} selected={selected} onSelect={onSelect} />
        ))}
      </ul>
    </nav>
  );
}
