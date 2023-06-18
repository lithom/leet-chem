package tech.molecules.leet.gui.chem.editor;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.JStructureView;
import tech.molecules.leet.chem.sar.SARDecompositionInstruction;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SARDecompositionFragmentList extends JPanel {

    /**
     * Model
     */
    public static class SARDecompositionFragmentListModel {
        private List<SARDecompositionInstruction.MultiFragment> decomposition = new ArrayList<>();
    }

    private SARDecompositionFragmentListModel model;

    public SARDecompositionFragmentList() {
        this.model = new SARDecompositionFragmentListModel();
        reinit();
    }

    public SARDecompositionFragmentList getThisComponent() {
        return this;
    }
    private void reinit() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        JPanel ptop = new JPanel(); ptop.setLayout(new FlowLayout(FlowLayout.RIGHT));
        JButton jb_add = new JButton("+");
        ptop.add(jb_add);
        this.add(ptop,BorderLayout.NORTH);

        JPanel plist = new JPanel();
        plist.setLayout(new GridLayout(1,model.decomposition.size()));
        for(int zi=0;zi<model.decomposition.size();zi++) {
            plist.add(new JSARDecompositionMultiFragment(model.decomposition.get(zi)));
        }
        this.add(plist,BorderLayout.CENTER);
        //this.invalidate();
        //this.repaint();

        SwingUtilities.updateComponentTreeUI(getThisComponent());

        jb_add.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                model.decomposition.add(new SARDecompositionInstruction.MultiFragment());
                reinit();
                //repaint();
            }
        });
    }

    public static class JSARDecompositionMultiFragmentElement extends JPanel {
        // Model
        private final SARDecompositionInstruction.MultiFragmentElement fi;

        public JSARDecompositionMultiFragmentElement(SARDecompositionInstruction.MultiFragmentElement fi) {
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
            JStructureView jcv = new JStructureView(fi.getFi());
            jcv.setMaximumSize(new Dimension(200,200));
            jcv.setBorder(new LineBorder(Color.black,1));
            this.add(jcv,BorderLayout.CENTER);
            this.setPreferredSize(new Dimension(200,200));
            this.setMinimumSize(new Dimension(200,50));
            //this.revalidate();
            //this.repaint();
            SwingUtilities.updateComponentTreeUI(this);
        }
    }


    public static class JSARDecompositionMultiFragment extends JPanel {
        // Model
        private final SARDecompositionInstruction.MultiFragment mfi;

        public JSARDecompositionMultiFragment(SARDecompositionInstruction.MultiFragment mfi) {
            this.mfi = mfi;
            reinitMF();
        }
        private void reinitMF() {
            this.removeAll();
            JPanel ptop = new JPanel();
            JButton jb_add = new JButton("+");
            //JButton jb_del  = new JButton("x");
            ptop.setLayout(new FlowLayout(FlowLayout.RIGHT));
            ptop.add(jb_add);//ptop.add(jb_del);
            //this.setLayout(new VerticalFlowLayout(VerticalFlowLayout.LEFT,VerticalFlowLayout.TOP));
            this.setLayout(new GridLayout(mfi.getMultiFragmentElements().size()+1,1));
            this.add(ptop);
            for(SARDecompositionInstruction.MultiFragmentElement fi : mfi.getMultiFragmentElements()) {
                this.add(new JSARDecompositionMultiFragmentElement(fi));
            }

            this.setMinimumSize(new Dimension(200,400));
            this.setPreferredSize(new Dimension(200,400));
            this.revalidate();
            this.repaint();


            jb_add.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    Random ri = new Random();
                    mfi.getMultiFragmentElements().add(new SARDecompositionInstruction.MultiFragmentElement("fid_"+ri.nextInt(1000),new StereoMolecule()));
                    reinitMF();
                    repaint();
                }
            });
        }
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

        SARDecompositionFragmentList list = new SARDecompositionFragmentList();
        fi.getContentPane().add(list,BorderLayout.SOUTH);
    }


}
