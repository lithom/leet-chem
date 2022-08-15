package tech.molecules.leet.table.gui;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.JStructureView;
import net.mahdilamb.colormap.Colormap;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;

import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class JStructureGridPanel extends JPanel {

    public JStructureGridPanel(List<String> idcs, int x, int y) {
        this.setData(idcs, x, y,null, new ArrayList<>(),new ArrayList<>());
    }

    public List<String> getSelected() {
        List<String> la = this.panels.stream().filter( pi -> pi.isSelected() ).map( pi -> pi.getStructure() ).collect(Collectors.toList());
        return new ArrayList<>(la);
    }

    public JStructureGridPanel(List<String> idcs, int x, int y, Map<String,List<Color>> color_annotations, List<Pair<String,Map<String,String>>> annotations, List<Triple<String, Map<String,Double>, Colormap>> numerical_data) {
        this.setData(idcs, x, y,color_annotations,annotations,numerical_data);
    }

//    @Override
//    protected void paintComponent(Graphics g) {
//        Color ca = new Color(254,254,254);
//        Color cb = new Color(246,251,251);
//        GradientPaint gp = new GradientPaint(0,0,ca,getWidth(),getHeight(),cb);
//        Graphics2D g2 = (Graphics2D) g;
//        g2.setPaint(gp);
//        g2.fillRect(0,0,this.getWidth(),this.getHeight());
//        super.paintComponent(g);
//    }

    private List<JGridPanel> panels = new ArrayList<>();

    public JPanel getThisJPanel() {return this;}

    public void setData(List<String> idcs, int x, int y, Map<String,List<Color>> color_annotations, List<Pair<String,Map<String,String>>> annotations, List<Triple<String, Map<String,Double>, Colormap>> numerical_data) {
        this.removeAll();
        IDCodeParser icp = new IDCodeParser();
        this.setLayout(new GridLayout(x, y));
        this.panels.clear();
        for (int zi = 0; zi < Math.min(idcs.size(), x * y); zi++) {
            StereoMolecule mi = new StereoMolecule();
            try {
                icp.parse(mi, idcs.get(zi));
            } catch (Exception ex) {
                System.out.println("[ERROR] problem with idcode?..");
            }
            JStructureView jva = new JStructureView(mi);
            //jva.setOpaque(false);

            jva.setBackground(new Color(255,255,255,0));
            JGridPanel gp = new JGridPanel(this,idcs.get(zi),jva);

            if(color_annotations!=null) {
                List<Color> lc2 = color_annotations.get(idcs.get(zi));
                if( lc2!=null && lc2.size()>0) {
                    //Color c2 = lc2.get(0);
                    //Color c2p = c2.brighter().brighter();
                    //Color c2pt = new Color(c2p.getRed(),c2p.getGreen(),c2p.getBlue(),30);
                    //jva.setBackground(c2pt);
                    gp.setBackgroundColors(lc2);
                }
            }

            this.panels.add(gp);
            this.add(gp);
            String fsi = idcs.get(zi);
            jva.addMouseListener(new MouseAdapter(){
                @Override
                public void mouseClicked(MouseEvent e) {
                    setMouseOverStructure(fsi);
                    if(e.getButton()==MouseEvent.BUTTON3) {
                        //getComponentPopupMenu().show(getThisJPanel(),e.getX(),e.getY());
                        myComponentPopupMenu.show(jva,e.getX(),e.getY());
                    }
                    else {
                        gp.toggleSelection();
                    }
                }

                @Override
                public void mouseEntered(MouseEvent e) {
                    System.out.println("mouse entered "+gp);
                    //setMouseOverStructure(fsi);
                    //gp.setMouseOver(true);
                }

                @Override
                public void mouseExited(MouseEvent e) {
                    //System.out.println("mouse exited "+gp);
                    //gp.setMouseOver(false);
                }
            });

            gp.setComponentPopupMenu(this.getComponentPopupMenu());
        }
        this.revalidate();
    }

    //private String structureMouseOver = null;

    /**
     *
     * @return null if mouse not over component
     */
    public String getStructureMouseOver() {
        return this.mouseOverStructure;
//        for(JGridPanel pa : this.panels) {
//            if(pa.isMouseOver()) {
//                return pa.getStructure();
//            }
//        }
//        return null;
    }


    private JPopupMenu myComponentPopupMenu;
    public void setContextMenu(JPopupMenu jpop) {
        this.myComponentPopupMenu = jpop;
        // this does not work, as it consumes the right click mouse events
        // that we need for identifying the mouseOver structure..
        //this.setComponentPopupMenu(jpop);
        //this.panels.stream().forEach( pi -> pi.setComponentPopupMenu(jpop) );
    }


    private String mouseOverStructure;
    private void setMouseOverStructure(String s) {
        this.mouseOverStructure = s;
    }

    public static class JGridPanel extends JPanel {

        private JStructureGridPanel parent;
        private String structure;
        private boolean selected = false;
        //private boolean mouseOver = false;
        private JComponent component;
        public JGridPanel(JStructureGridPanel parent, String structure, JComponent c) {
            this.parent = parent;
            this.structure = structure;
            this.component = c;
            this.setLayout(new BorderLayout());
            this.add(c,BorderLayout.CENTER);

            // add annotation panels:


            this.addMouseListener( new MouseListener() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    parent.setMouseOverStructure(structure);
                }

                @Override
                public void mousePressed(MouseEvent e) {
                    parent.setMouseOverStructure(structure);
                    if(e.getButton()==MouseEvent.BUTTON3 && isSelected() ) {
                        getComponentPopupMenu().show(getThisJPanel(),e.getX(),e.getY());
                    }
                    else {
                        toggleSelection();
                    }
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                }
                @Override
                public void mouseEntered(MouseEvent e) {
                    //setMouseOver(true);
                    //setMouseOverStructure(structure);
                    //System.out.println("mouse over: "+this);
                }

                @Override
                public void mouseExited(MouseEvent e) {
                    //setMouseOver(false);
                    //System.out.println("mouse exited: "+this);
                }
            } );
        }

        @Override
        protected void paintComponent(Graphics g_) {
            super.paintComponent(g_);
            Graphics2D g = (Graphics2D) g_;
            // draw background:
            if(this.backgroundColors!=null && this.backgroundColors.size()>0) {
                int nbc = this.backgroundColors.size();
                if(nbc==1) {
                    Color ca_1 = backgroundColors.get(0);
                    Color ca = new Color(ca_1.getRed(),ca_1.getGreen(),ca_1.getBlue(),40);
                    Color cb = new Color(ca_1.getRed(),ca_1.getGreen(),ca_1.getBlue(),100);
                    g.setPaint(new GradientPaint(0,0,ca,getWidth(),getHeight(),cb));
                    g.fillRect(0,0,getWidth(),getHeight());
                }
                else {
                    for(int zi=0;zi<nbc;zi++) {
                        Color ca_1 = backgroundColors.get(zi);
                        Color ca = new Color(ca_1.getRed(),ca_1.getGreen(),ca_1.getBlue(),40);
                        Color cb = new Color(ca_1.getRed(),ca_1.getGreen(),ca_1.getBlue(),100);
                        int pxa = (int) ( zi* ( (1.0*getWidth()) / nbc ) );
                        int pxb = (int) ( (zi+1)* ( (1.0*getWidth()) / nbc ) );
                        g.setPaint(new GradientPaint(pxa,0,ca,pxb,this.getHeight(),cb));
                        g.fillRect(pxa,0,(pxb-pxa),this.getHeight());
                    }
                }
            }
            else {
                Color ca = Color.white;
                Color cb = new Color(245,250,250);
                g.setPaint(new GradientPaint(0,0,ca,this.getWidth(),this.getHeight(),cb));
                g.fillRect(0,0,this.getWidth(),this.getHeight());
            }

        }

        public JPanel getThisJPanel() {return this;}

        @Override
        public void setComponentPopupMenu(JPopupMenu popup) {
            super.setComponentPopupMenu(popup);
            this.component.setComponentPopupMenu(popup);
        }

        public void select(boolean selected) {
            this.selected = selected;
            if(this.selected) {
                this.setBorder(new LineBorder(Color.red.darker(),2));
            }
            else{
                this.setBorder(new LineBorder(Color.black,1));
            }
            this.repaint();
        }

        public void toggleSelection() {
            this.select(!this.selected);
        }

        public boolean isSelected() { return this.selected; }

        //public boolean isMouseOver() { return this.mouseOver; }
        public String  getStructure() {return this.structure; }

        private List<Color> backgroundColors;
        public void setBackgroundColors(List<Color> ci) {
            this.backgroundColors = new ArrayList<>(ci);
        }

        //public void setMouseOver(boolean mouseOver) {
        //    this.mouseOver = mouseOver;
        //}
    }
}
