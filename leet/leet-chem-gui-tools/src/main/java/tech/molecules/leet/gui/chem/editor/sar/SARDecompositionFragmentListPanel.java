package tech.molecules.leet.gui.chem.editor.sar;

import tech.molecules.leet.chem.sar.MultiFragment;
import tech.molecules.leet.chem.sar.SARDecompositionInstruction;
import tech.molecules.leet.chem.sar.SARElement;
import tech.molecules.leet.chem.sar.SimpleMultiFragment;
import tech.molecules.leet.gui.chem.editor.SARDecompositionEditor;
import tech.molecules.leet.gui.chem.editor.SARDecompositionFragmentList;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

public class SARDecompositionFragmentListPanel extends JPanel implements SARDecompositionFragmentListModel.MultiFragmentListListener {

    private SARDecompositionFragmentListModel model;

    public SARDecompositionFragmentListPanel(SARDecompositionFragmentListModel model) {
        this.model = model;

        model.addMultiFragmentListListener(this);
        reinit();
    }

    public void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());
        JPanel p_list = new JPanel();
        p_list.setLayout(new GridLayout(1,model.getMultiFragments().size()));
        for(SimpleMultiFragment fi : this.model.getMultiFragments()) {
            p_list.add(new MultiFragmentPanel(fi));
        }
        this.add(p_list,BorderLayout.CENTER);

        JPanel ptop = new JPanel();
        ptop.setLayout(new FlowLayout(FlowLayout.RIGHT));
        JButton b_add = new JButton("+");
        ptop.add(b_add);
        b_add.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                model.addMultiFragment(new SimpleMultiFragment());
            }
        });
        this.add(ptop,BorderLayout.NORTH);
        //this.revalidate();
        SwingUtilities.updateComponentTreeUI(this);
    }

    @Override
    public void onMultiFragmentAdded(SimpleMultiFragment multiFragment) {
        this.reinit();
    }

    @Override
    public void onMultiFragmentRemoved(int index) {
        this.reinit();
    }

    @Override
    public void onSARElementSelected() {

    }

    public static void main(String args[]) {
        JFrame fi = new JFrame();
        fi.setSize(600,600);

        SARDecompositionEditor editor = new SARDecompositionEditor();
        editor.setFocusable(true);
        editor.setRequestFocusEnabled(true);
        fi.getContentPane().setLayout(new BorderLayout());
        fi.getContentPane().add(editor,BorderLayout.CENTER);
        fi.setVisible(true);

        SARDecompositionFragmentListModel model = new SARDecompositionFragmentListModel();
        SARDecompositionFragmentListPanel list = new SARDecompositionFragmentListPanel(model);
        fi.getContentPane().add(list,BorderLayout.SOUTH);
    }





}
