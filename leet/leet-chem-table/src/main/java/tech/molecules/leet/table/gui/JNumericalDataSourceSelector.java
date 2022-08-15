package tech.molecules.leet.table.gui;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import tech.molecules.leet.table.NColumn;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.NumericalDatasource;

import javax.swing.*;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class JNumericalDataSourceSelector  extends JPanel {

    public static class NumericalDataSourceSelectorModel {
        private NexusTableModel ntm;
        public NumericalDataSourceSelectorModel(NexusTableModel ntm) {
            this.ntm = ntm;
        }
        public DefaultTreeModel getTreeModel() {
            DefaultMutableTreeNode root = new DefaultMutableTreeNode();
            Map<NColumn, Map<String, NumericalDatasource>> cols = this.ntm.collectNumericDataSources();
            for(NColumn ci : cols.keySet().stream().sorted( (x,y) -> x.getName().compareTo(y.getName()) ).collect(Collectors.toList()) ) {
                DefaultMutableTreeNode ncol = new DefaultMutableTreeNode(ci.getName());
                root.add(ncol);
                for(String si : cols.get(ci).keySet().stream().sorted().collect(Collectors.toList())) {
                    ncol.add( new DefaultMutableTreeNode( cols.get(ci).get(si) ) );
                }
            }
            DefaultTreeModel dtm = new DefaultTreeModel(root);
            return dtm;
        }

        private NumericalDatasource selectedDatasource;

        public void setSelectedDatasource(NumericalDatasource nd) {
            this.selectedDatasource = nd;
        }

        public NumericalDatasource getSelectedDatasource() {
            return this.selectedDatasource;
        }

    }

    private NumericalDataSourceSelectorModel model;

    public enum SELECTOR_MODE {Tree,Menu};
    public JNumericalDataSourceSelector(NumericalDataSourceSelectorModel model, SELECTOR_MODE mode) {
        this.model = model;

        if(mode.equals(SELECTOR_MODE.Tree)) {
            initTree();
        }
        else if(mode.equals(SELECTOR_MODE.Menu)) {
            initMenu();
        }
    }

    private JScrollPane jsp;
    private JTree jt;

    private JToolBar jtb;
    private JButton jtb2;
    private JMenu    jm;

    private void initTree() {
        this.removeAll();
        this.jsp = new JScrollPane();
        this.setLayout(new BorderLayout());
        this.add(this.jsp,BorderLayout.CENTER);
        this.jt = new JTree();
        if(this.model.getTreeModel()!=null) {
            this.jt.setModel(this.model.getTreeModel());
        }
        this.jsp.setViewportView(this.jt);

        jt.addTreeSelectionListener(new TreeSelectionListener() {
            @Override
            public void valueChanged(TreeSelectionEvent e) {
                if( ((DefaultMutableTreeNode)jt.getLastSelectedPathComponent()).isLeaf()) {
                    model.setSelectedDatasource((NumericalDatasource) ((DefaultMutableTreeNode) jt.getLastSelectedPathComponent()).getUserObject());
                }
            }
        });
    }

    public JPopupMenu getMenu() {
        return this.jm.getPopupMenu();
    }
    private void initMenu() {
        this.removeAll();
        this.setLayout(new BorderLayout());
        //this.jtb = new JToolBar();
        //this.add(this.jtb,BorderLayout.CENTER);
        this.jm = null;
        JMenu ji = new JMenu();
        addChildrenToMenu(ji,((DefaultMutableTreeNode)model.getTreeModel().getRoot()));
        //this.jm.add(ji);
        this.jm = ji;

        this.jtb2 = new JButton("Set NDS");
        this.jtb2.setComponentPopupMenu(this.jm.getPopupMenu());
//        this.jtb2.addActionListener(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                jm.getPopupMenu().show
//            }
//        });

        this.add(jtb2);
    }

    private class SelectAction extends AbstractAction {
        private NumericalDatasource ndi;
        public SelectAction(NumericalDatasource ndi) {
            super(ndi.getName());
            this.ndi = ndi;
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            model.setSelectedDatasource(ndi);
            fireSelectionChangedEvent();
        }
    }
    private void addChildrenToMenu(JMenuItem jmi, DefaultMutableTreeNode dmtn) {
        for(int zi=0; zi< dmtn.getChildCount();zi++ ) {
            DefaultMutableTreeNode ni = ((DefaultMutableTreeNode)dmtn.getChildAt(zi));
            if(ni.isLeaf()) {
                ((JMenu)jmi).add(new SelectAction((NumericalDatasource) ni.getUserObject()));
            }
            else {
                JMenu jmnew = new JMenu(ni.getUserObject().toString() );
                jmi.add(jmnew);
                addChildrenToMenu(jmnew,ni);
            }
        }
    }


    public static interface SelectionListener {
        public void selectionChanged();
    }

    private List<SelectionListener> selListeners = new ArrayList<>();
    public void addSelectionListener(SelectionListener li) {
        this.selListeners.add(li);
    }
    private void fireSelectionChangedEvent() {
        for(SelectionListener li : this.selListeners) {
            li.selectionChanged();
        }
    }

//    public void addActionListenerToMenu(ActionListener li) {
//        addActionListenerToMenu_rec(li,null);
//    }
//    private void addActionListenerToMenu_rec(ActionListener li,JMenuItem mi) {
//        if(mi==null) {
//            addActionListenerToMenu_rec(li,this.jm.getItem(0));
//            return;
//        }
//        mi.addActionListener(li);
//        if(mi instanceof JMenu) {
//            for (int zi = 0; zi < ((JMenu)mi).getItemCount(); zi++ ) {
//                addActionListenerToMenu_rec(li,((JMenu)mi).getItem(zi));
//            }
//        }
//    }

    public static Triple<JPanel, Supplier<NumericalDatasource>, Consumer<NumericalDatasource>> getSelectorPanel2(NexusTableModel ntm, Frame owner) {
        JPanel pi = new JPanel();
        pi.setLayout(new FlowLayout(FlowLayout.LEFT));
        JTextField jti = new JTextField(18);
        jti.setText("");

        //JButton jb = new JButton("Set Numerical DS");

        NumericalDataSourceSelectorModel ndsm = new NumericalDataSourceSelectorModel(ntm);

        // react to ok button:
        Supplier<NumericalDatasource> nds_supplier = () -> ndsm.getSelectedDatasource();

        JNumericalDataSourceSelector jndsm    = new JNumericalDataSourceSelector(ndsm,SELECTOR_MODE.Menu);
        pi.add(jti);
        pi.add(jndsm);
        //pi.add(jb);

        Consumer<NumericalDatasource> nds_consumer = (nds) -> jti.setText(nds.getName());
        jndsm.addSelectionListener(new SelectionListener() {
            @Override
            public void selectionChanged() {
                jti.setText(ndsm.getSelectedDatasource().toString());
            }
        });
//        jndsm.addActionListenerToMenu(new ActionListener() {
//            @Override
//            public void actionPerformed(ActionEvent e) {
//                jti.setText(ndsm.getSelectedDatasource().toString());
//            }
//        });

        return Triple.of(pi,nds_supplier,nds_consumer);
    }

    public static Triple<JPanel, Supplier<NumericalDatasource>, Consumer<NumericalDatasource>> getSelectorPanel(NexusTableModel ntm, Frame owner) {
        JPanel pi = new JPanel();
        pi.setLayout(new FlowLayout(FlowLayout.LEFT));
        JTextField jti = new JTextField(18);
        jti.setText("");

        JButton jb = new JButton("Set Numerical DS");

        NumericalDataSourceSelectorModel ndsm = new NumericalDataSourceSelectorModel(ntm);

        // react to ok button:
        Supplier<NumericalDatasource> nds_supplier = () -> ndsm.getSelectedDatasource();

        jb.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JDialog jd = new JDialog(owner,true);

                JNumericalDataSourceSelector jndsm    = new JNumericalDataSourceSelector(ndsm,SELECTOR_MODE.Tree);
                jndsm.setPreferredSize(new Dimension(200,320));

                jd.getContentPane().setLayout(new BorderLayout());
                jd.getContentPane().add(jndsm,BorderLayout.CENTER);

                JPanel bottom = new JPanel();
                bottom.setLayout(new FlowLayout(FlowLayout.RIGHT));
                JButton jok = new JButton("OK");
                bottom.add(jok);
                jd.getContentPane().add(bottom,BorderLayout.SOUTH);

                jok.addActionListener(new ActionListener() {
                    @Override
                    public void actionPerformed(ActionEvent e) {
                        jti.setText(ndsm.getSelectedDatasource().getName());
                        jd.setVisible(false);
                    }
                });
                jd.pack();
                jd.setVisible(true);
            }
        });

        pi.add(jti);
        pi.add(jb);

        Consumer<NumericalDatasource> nds_consumer = (nds) -> jti.setText(nds.getName());

        return Triple.of(pi,nds_supplier,nds_consumer);
    }

    public static Pair<JButton, Supplier<NumericalDatasource>> getAccessButton(NexusTableModel ntm, Frame owner) {
        JButton jb = new JButton("Set Numerical DS");

        NumericalDataSourceSelectorModel ndsm = new NumericalDataSourceSelectorModel(ntm);

        // react to ok button:
        Supplier<NumericalDatasource> nds_supplier = () -> ndsm.getSelectedDatasource();

        jb.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JDialog jd = new JDialog(owner,true);

                JNumericalDataSourceSelector jndsm    = new JNumericalDataSourceSelector(ndsm,SELECTOR_MODE.Tree);

                jd.getContentPane().setLayout(new BorderLayout());
                jd.getContentPane().add(jndsm,BorderLayout.CENTER);

                JPanel bottom = new JPanel();
                bottom.setLayout(new FlowLayout(FlowLayout.RIGHT));
                JButton jok = new JButton("OK");
                bottom.add(jok);
                jd.getContentPane().add(bottom,BorderLayout.SOUTH);

                jok.addActionListener(new ActionListener() {
                    @Override
                    public void actionPerformed(ActionEvent e) {
                        jd.setVisible(false);
                    }
                });
                jd.pack();
                jd.setVisible(true);
            }
        });
        return Pair.of(jb,nds_supplier);
    }

}
