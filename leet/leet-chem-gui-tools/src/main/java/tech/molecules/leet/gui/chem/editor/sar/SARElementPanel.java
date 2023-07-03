package tech.molecules.leet.gui.chem.editor.sar;

import com.actelion.research.gui.JStructureView;
import tech.molecules.leet.chem.sar.MultiFragment;
import tech.molecules.leet.chem.sar.SARDecompositionInstruction;
import tech.molecules.leet.chem.sar.SARElement;
import tech.molecules.leet.chem.sar.SimpleSARElement;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

public class SARElementPanel extends JPanel {



    // Model
    private final SimpleSARElement fi;

    public SARElementPanel(SimpleSARElement fi) {
        this.fi = fi;
        initMFE();
    }
    private void initMFE() {
        this.removeAll();
        this.setLayout(new BorderLayout());
        JPanel ptop = new JPanel();
        this.add(ptop,BorderLayout.NORTH);
        JButton jb_edit = new JButton("e");
        JButton jb_del  = new JButton("x");
        ptop.setLayout(new FlowLayout(FlowLayout.RIGHT));
        ptop.add(jb_edit);ptop.add(jb_del);
        //JEditableStructureView jcv = new JEditableStructureView(fi.getFi());
        JStructureView jcv = new JStructureView(fi.getMol());
        jcv.setMaximumSize(new Dimension(200,200));
        jcv.setBorder(new LineBorder(Color.black,1));
        this.add(jcv,BorderLayout.CENTER);
        this.setPreferredSize(new Dimension(200,200));
        this.setMinimumSize(new Dimension(200,50));
        //this.revalidate();
        //this.repaint();

        jb_edit.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                fireOnEdit();
            }
        });
        jb_del.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                fireOnRemove();
            }
        });

        SwingUtilities.updateComponentTreeUI(this);
    }

    public static interface SARElementListener {
        public void onEdit();
        public void onRemove();
    }

    private void fireOnEdit() {
        for(SARElementListener li : listeners) {
            li.onEdit();
        }
    }
    private void fireOnRemove() {
        for(SARElementListener li : listeners) {
            li.onRemove();
        }
    }


    private List<SARElementListener> listeners = new ArrayList<>();

    public void addSARElementListener(SARElementListener listener) {
        listeners.add(listener);
    }

    public void removeSARElementListener(SARElementListener listener) {
        listeners.remove(listener);
    }


}
