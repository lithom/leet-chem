package tech.molecules.leet.gui.chem.project;

import tech.molecules.leet.commons.tree.AbstractMappedTreeNode;

import javax.swing.*;
import javax.swing.tree.TreeModel;
import javax.swing.tree.TreePath;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class SeriesManagerTreeView extends JPanel {

    private TreeModel treeModel;
    private ProjectTreeController treeController;

    private JScrollPane scrollPane;
    private JTree       tree;

    public SeriesManagerTreeView(TreeModel treeModel, ProjectTreeController treeController) {
        this.treeModel = treeModel;
        this.treeController = treeController;

        this.reinit();
    }

    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.tree = new JTree(treeModel);
        this.scrollPane = new JScrollPane(this.tree);
        this.add(scrollPane,BorderLayout.CENTER);

        // Add MouseListener for context menu
        tree.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (SwingUtilities.isRightMouseButton(e)) {
                    TreePath path = tree.getPathForLocation(e.getX(), e.getY());
                    if (path != null) {
                        AbstractMappedTreeNode<?> node = (AbstractMappedTreeNode<?>) path.getLastPathComponent();
                        JPopupMenu contextMenu = treeController.createContextMenu(node);
                        if (contextMenu != null) {
                            contextMenu.show(tree, e.getX(), e.getY());
                        }
                    }
                }
            }
        });

    }



}
