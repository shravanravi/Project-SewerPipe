function Ai = generate_Ai(m_i);
    I = eye(size(m_i , 1));
    Ai = I(find(not(m_i)),:);
return;
