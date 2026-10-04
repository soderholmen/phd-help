// The section tree (SPEC §1): clicking a node anchors the section-scoped
// discussion; the anchor is clearable (select_section with "").
import type { SectionNode } from "../types";

interface NodeProps {
  node: SectionNode;
  selected: string | null;
  onSelect: (path: string) => void;
}

function Node({ node, selected, onSelect }: NodeProps) {
  // Indent is the nested ul's own margin (CSS), not per-node padding:
  // the text stays a direct child of the button, and the guide line
  // lands under the parent's label for free.
  return (
    <li>
      <button
        className="tree-node"
        aria-current={selected === node.path ? "true" : undefined}
        onClick={() => onSelect(node.path)}
      >
        {node.title ?? node.path}
      </button>
      {node.children.length > 0 && (
        <ul>
          {node.children.map((c) => (
            <Node key={c.path} node={c} selected={selected} onSelect={onSelect} />
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
      {tree.length === 0 ? (
        <p className="tree-empty">no sections yet</p>
      ) : (
        <ul>
          {tree.map((n) => (
            <Node key={n.path} node={n} selected={selected} onSelect={onSelect} />
          ))}
        </ul>
      )}
    </nav>
  );
}
