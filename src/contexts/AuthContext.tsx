import React, { createContext, useContext, useState, ReactNode } from 'react';

interface User {
  email: string;
  username: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, username: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  const login = async (email: string, password: string) => {
    // Mock login - in production, this would call an API
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Extract username from email for mock purposes
    const username = email.split('@')[0];
    setUser({ email, username });
  };

  const signup = async (email: string, password: string, username: string) => {
    // Mock signup - in production, this would call an API
    await new Promise(resolve => setTimeout(resolve, 800));
    setUser({ email, username });
  };

  const logout = () => {
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        login,
        signup,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
