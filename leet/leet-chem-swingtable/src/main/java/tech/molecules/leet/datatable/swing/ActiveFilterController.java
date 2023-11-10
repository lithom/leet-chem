package tech.molecules.leet.datatable.swing;

import com.actelion.research.util.BrowserControl;
import tech.molecules.leet.datatable.DataTable;

import javax.swing.*;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeModel;
import java.awt.*;
import java.util.List;

public class ActiveFilterController extends JPanel {

    private DefaultSwingTableController table;

    // GUI:

    private JPanel pTop;
    private JPanel pMain;
    private JPanel pMainLeft;
    private JPanel pMainRight;

    private JScrollPane jspMain;
    private JTree jtMain;

    public ActiveFilterController(DefaultSwingTableController table) {
        this.table = table;

        this.table.getModel().getDataTable().addDataTableListener(new DataTable.DataTableListener() {
            @Override
            public void tableDataChanged() {
                reinitData();
            }

            @Override
            public void tableStructureChanged() {
                reinit();
            }

            @Override
            public void tableCellsChanged(List<int[]> cells) {
                // might be because filtering changed
                reinitData();
            }
        });
        this.reinit();
    }

    private void reinit() {
        this.currentTreeModel = null;
        reinitGUI();
        reinitData();
    }

    private void reinitGUI() {

        // GUI:
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.pTop  = new JPanel();
        this.pMain = new JPanel(); this.pMain.setLayout(new BorderLayout());

        this.add(pMain,BorderLayout.CENTER);
        this.pMainLeft  = new JPanel(); this.pMainLeft.setLayout(new BorderLayout());
        this.pMainRight = new JPanel();
        this.pMain.add(this.pMainLeft,BorderLayout.CENTER);

        this.jspMain = new JScrollPane();
        this.jtMain  = new JTree();
        this.jspMain.setViewportView(this.jtMain);

        this.pMainLeft.add(this.jspMain,BorderLayout.CENTER);
    }

    private TreeModel currentTreeModel = null;

    private void reinitData() {
        TreeModel newModel = table.createActiveFilterTreeModel();
        if( currentTreeModel==null || ! TreeComparison.compareTreeModels( (DefaultTreeModel) newModel,(DefaultTreeModel) this.currentTreeModel)) {
            this.jtMain.setModel(newModel);
            this.currentTreeModel = newModel;
            //this.jtMain.repaint();
            SwingUtilities.updateComponentTreeUI(this);
        }
    }


    public static class TreeComparison {

        public static boolean compareTreeModels(DefaultTreeModel treeModel1, DefaultTreeModel treeModel2) {
            DefaultMutableTreeNode root1 = (DefaultMutableTreeNode) treeModel1.getRoot();
            DefaultMutableTreeNode root2 = (DefaultMutableTreeNode) treeModel2.getRoot();

            return compareNodes(root1, root2);
        }

        private static boolean compareNodes(DefaultMutableTreeNode node1, DefaultMutableTreeNode node2) {
            if (node1 == null && node2 == null) {
                return true;
            } else if (node1 == null || node2 == null) {
                return false;
            }

            // Compare user objects (you need to implement this comparison logic)
            Object userObject1 = node1.getUserObject();
            Object userObject2 = node2.getUserObject();

            if(userObject1==null || userObject2==null) {
                if( !(userObject1==null && userObject2==null) ) {
                    return false;
                }
            }
            else {
                if (!userObject1.equals(userObject2)) {
                    return false;
                }
            }

            // Recursively compare child nodes
            int childCount1 = node1.getChildCount();
            int childCount2 = node2.getChildCount();

            if (childCount1 != childCount2) {
                return false;
            }

            for (int i = 0; i < childCount1; i++) {
                DefaultMutableTreeNode child1 = (DefaultMutableTreeNode) node1.getChildAt(i);
                DefaultMutableTreeNode child2 = (DefaultMutableTreeNode) node2.getChildAt(i);

                if (!compareNodes(child1, child2)) {
                    return false;
                }
            }

            return true;
        }

    }




}
